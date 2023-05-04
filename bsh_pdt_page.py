import streamlit as st
import pandas as pd
from io import StringIO
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests
from urllib.parse import quote,unquote
import requests
import re
import csv
from lxml import etree
import os
import time
import pandas as pd
from PIL import Image
import io
# import cv2
from urllib.parse import quote,unquote
from sentence_transformers import SentenceTransformer, util
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics.pairwise import cosine_similarity
import hypertools as hyp
import random
from paddleocr import PaddleOCR, draw_ocr# Paddleocr目前支持中英文、英文、法语、德语、韩语、日语，可以通过修改lang参数进行切换   # 参数依次为`ch`, `en`, `french`, `german`, `korean`, `japan`。
ocr = PaddleOCR(use_angle_cls=True, lang="ch") # need run only once to download and load model into memory
from keybert import KeyBERT
kw_model = KeyBERT()

############################################################################
dropword = ['【不良反应】','理性阅读','以上对比','油度检测','侧板固定','物品搬运','结构重构','用户自备','踢脚线','切割封板','水款标识','浊度检测','铰链调整','门板安装','安装建议','水效标识','一级标准','清洁指数','干燥指数','不得安装','点击咨询','底板开','家电家私','点击查看价格','联系方式','随机发送','官方旗舰店','售后', '服务', '京仓', '发货','仓配', '出库', '授权', '风险', '放心购', '补寄', '配送', '亲们', '仓离', '限时达', '免费补发', '先验货', '再签收','送货', '上门', 
            '破损', '包赔', '亲爱的顾客','调换货', '采购', '新老包装', '更替', '产品外观', '中文背标', '生产', '断货', '亲们', '多包涵', '客服', '发货', '开具', '发票', '八仓', 
            '相关规定','本店', '销售酒水', '自带', '加贴', '背标', '温馨提示', '下单', '参数','授权书', '检验', '检疫', '报关单', '未成年人', '请勿饮酒', '正品', '禁止孕妇饮酒', 
            '关于产品','实时监控','收货','部分地区','大仓','上午购买','质量保障','实际','关于质期','“酒”是快','关于保质期','为准','质检单','退换货','妥善给您解决问题'
           '根据省份','胶带','小哥','快递','签收','须知','货主','卫检','报关','摆放有序','京方有限公司','购放心','温馨提醒','关于赠品','酒后请勿驾','存放于','饮酒不可开车','北京天津河北山西',
            '过量饮酒','甘肃宁夏','请勿酒后''酒精含量','关于赠品','商品名称','商品规格','如何成为','未成年','立省','拒绝久等', '派送问题','产品信息','【警示用语】过量饮酒有害健康','头购国',
            '储藏条件','买惠多多','合饮用','产品实拍','见瓶身喷码','盗图','如图所示','见附页','国际贸易有限公司','报检','运输过程','饮用说明书','日期会有所偏差',
           '商家','退款','慎拍','包裹','物流','批次','饮酒友情提示','正面','背面','选择我们','挤压','仓库','寄回','发错货','置于阴凉','贮存方式','切勿懂击','销售门店',
            '批运单号','优布劳精','到您身','下江山','正院','美人龙井小','八箱','职业打假人','踩屎感2078','不单送起酒器','手机扫码','净含量','实拍','产品展示','避免震动','扫码验证',
           '硬质包装','场景展示','商品展示','存储方式','购买需知','容量为','所属公司','储藏与管理','有保障','年月日','北京卫视','有限公司','江苏镇江','注意事项','保质期限','商品详细',
            '运输包装','产品尺寸','安全负担','产品类型','营业执照','注册资本','单笔订单金额满','保质期','股份有限公司','质量体系认证','【规格】','储藏保存','避免阳光直射','【容量】',
            '避免阳光直射','耐热度℃℃','贮存条件','避免光照','恒温储藏','保持空气流通','储存湿度','净含费','公家证','始建年','业集团','格合量','自然不样','口惊喜','查询电话','登陆查询',
            '商品卖点','委托单位','营专区','年月日罗马','何时何地','正标展示','有限公司','分享●','商务有限公','即可查询','防伪查询','注购买单瓶装不含礼袋','酒业有限公司','防伪电话',
            '腾讯视频','净含业','爱奇艺','我们的优势','顾客买家秀','信赖之选','装运输箱','买家秀','厂商介绍','人人人人人人人人人人人','发公司','冰镇冷藏分钟后','加热饮用',
            '建议饮用温度','贮藏方式','台集生农业','防伪','开盖小心','品牌介绍','温禁提示','购买理由','酒业礼器','中华人民共和国商务部','使用扫码软件','产品好与坏','库存','锡纸',
            '朝向安全处','警数要求严格','全国十七大','单位','超值量贩装','搭配购买','左右适宜','使用金属网固定','注意喷射','适饮温度推荐','送礼贝】有面儿','醉别江楼橘柚香',
            '螺旋杆斜尖进入木塞中心','贵州惠水梯田海拔米','好的酒般都不贵','选择俏雅梅酒理由','不计成本只用梅中上品','西山本地当季青梅','狼小姐的甜酒铺','百利×碎冰摩卡',
            '狮子歌歌花椒草蚕纯果酿','酒时浪冷萃茶酒','浸人麦烧酒阁魔的原酒中','人于年建立了思慕客酒庄直坚持家族','思慕客萨摩罗德尼贵腐','每天次丰工翻浆','澳迪尼月夕桂花酿',
            '奈良果中西部第二阪奈道路办通大阪办','托尼斯帝芬尼亚','花田巷子桑费酒','梅酒飞体全国初','封家书纸短情号','の余分决添加为访力世凡','新时和间政路业有限强',
            '私博付物语仁飞酒水东石','丘心限造成就美酒','青梅煮酒斗时新','慕拉庄园园主展肖华点评','金赏安多数受赏下石','宠幸【西施】兴建【馆娃宫','尼业加拉大澡布尼业加拉还孕',
            '河武酸造二重票即在法国举办的年由女性主导的','李鸿章签发营业准照光绪皇帝老师翁同解先生为张裕题写厂名','狮子歌歌高阶系列','苏州桥与永辉超市盒马','工社長美人の版记层',
            '新训和间欢路业有限或任公司','成都平原天府之国','青缘起壶桃花醉','蚕歌真迹与现代','宁夏贺兰山东机','红杏初生叶青梅已经枝','悠日田の山育自然の惠为','滴美妙柔附苦尖',
            '底部材叙路可見','女儿红三年陈花雕','绣球花丹晚干红','日酸<後味办日','花筑身’以酒养命','幸福在甘咨小办拍石梅酒','花堪折桃紅起泡酒','游龙荣罐秋菊华茂春松','当看梅鲜泥米洒',
            '私大古注酒苏物语','梅甘梅酒老完熟己世面','溪用泽自和歌山照','单丛蜜桃茶果酒','葡小萄草莓西柚','宫草莓果肉果酒','托卡伊签奥苏贵腐甜葡萄酒精华','颗水果酿造瓶果酒',
            '萄酿制瓶葡萄酒','格哈古雷斯草莓奶油利口酒','酿造无添脱醇工艺','原料桂花糯米山泉水酒曲','梨汁加气起泡酒','酒庄灌装原瓶进口','酒威士忌梅酒宇治茶起泡梅酒','正蜂蜜发酵的果酒',
            '原生水果发酵型果酒','苏榴兰单麦芽威士忌','微熏甜酒梅子酒','果肉梅酒梅酒组合','卡斐娜波萝椰奶味利口酒','阿斯蒂高泡甜白起泡酒','≥果汁青梅酿造',
            '微西美系列果酒','本格梅酒除使用梅子糖类','瓶毫升的青梅酒','拒绝勾兑颗水果酿造瓶果酒','知熏轻酸奶酒系列','飛露喜威士忌梅酒','以清酒工艺标准酸制基酒',
            '户河内樱桃味威士忌配制酒','青梅冰糖百花蜜酿造','百利甜酒×热生奶','本格梅酒是只使用梅子糖基酒','除了梅砂糖基酒','约三斤水蜜桃酿壶酒','名杯莱尼诺查韵日葡萄起泡酒',
            '荷菲桃红起泡酒','桃红莫斯卡托起泡酒','德国雷司令甜白金奖贵腐酒','天柔红山葡萄甜酒','剔核纯梅肉发酵','成茶府的县京都到茶的产绿','家死硫甜酒的店铺','原味精限担绝勾完',
            '馆娃宫桂花米露','馆娃宫蜜桃米露','戴夕桃红起泡酒火烈鸟桃红','赋比兴芽茶梅酒','天满天神梅酒大会','野格圣鹿利口酒','江小果红动果酒','梅乃宿果肉桃酒',
            '是正常现象喝前请摇晃均匀哦','她们融为体甜梦今夜','与尔同销万古愁','西楼无客共准尝','拍照不修也出圈','杯不畏将来不念过去做只','月了然处处自然','粉繁熙接的夏日裹',
            '四季中浪海不过晚春与初息','引这也是为什么猫脱第款是','从小闻得题多的就是桂','想把这份说话甘座党','东关住夜夜湖中看月生','出地球环游星际','狮子歌歌柚子梅酒微微冰',
            '每瓶雀古春见里的丑橘发酵酒','与贵腐酒总量的左石血','魔免子蒸馅小酒','八木酒造×青短柚子酒','百年酒造老松酒厂','茶藏红花东方香料与些许檀查查气陈年贵腐的温季',
            '玫瑰野花相橘菠萝','步小野梅酒神仙吃法','女士梅子酒晚安利口酒','清澈的门酒用水和傻查造酒好米加上杜氏的傅统技術所','利白年蒸馅工艺酿就出了皮斯科基','日本国内的本格烧酒梅酒食用王隐堂农','五年陈库藏黄酒','酿造瓶好酒分靠原料',
            '桂满麓桂花冰酿','惠比寿福梅子酒','仁蜂蜜葡萄干冰糖','待魂萄气果味葡萄酒','材料梅砂糖基酒','水蜜桃汁水发酵酒杨梅汁',
            '葡萄酒般使用范菜红晒脂红焦糖色香精酒精甜蜜素等色素加水勾兑而成遇碱不会发生','瓶白洋河冰纯浓缩葡萄酒',
            '鲜果精酿红西柏酒','蜂蜜柚子果泥酒蜂蜜山植果泥酒','蜂蜜山楂果泥酒','蜂蜜柚子果泥酒配制酒','成都蒲江县丑橘','万岁乐加贺梅酒','洛菲黑魅天鹅冰白','关干创牙利国家馆','名工区京产三良人',
            '南高梅梅使用量','雅协调尾净味长空杯留','用第级翘起木塞','名字时客桂花酒','需冻隆临深秋的果园','貂典黄酒女儿红','雷可令晚收半甜白','六入>>梅子の','叶槽版群口相声','薄荷叶片莫罗号酒',
            '慕拉雷司令蓝钻葡萄酒','金兹玛拉乌利红葡萄酒','牙利国酒液体黄金','五女山红冰葡萄酒','和闻红萃石榴酒','玲珑草莓冻撞酒','眼造混合威士忌和单纯麦威士局',
            '浸米蒸饭淋饭拌曲发酵出酒等道古法酿造工序才得以产生','橙汁俏雅梅酒=','户河内梅子味威士忌','十年熟成本格梅酒','酒庄拥有牙利属香贵腐的华断性产量','高千穗熟成梅酒','电芝野生蓝莓酒','张裕贵馥晚采甜白葡萄酒','全少编含看朵花瓣的精华','成都蒲江县雀舌','棵树只酿瓶好酒','万晟染酸造酒店万晟染','专用瓦榜纸外箱','香芋倍加忠于冷暖紫知的本味','符猎者常见喝法','微酶快意如酿刚刚好','甸牙利驻华大便馆晚宴用酒','日本法裂造礼盒款去寸','帕拉丁特干年份普罗塞克','佰蜜创意瓶头国潮玩偶','梅否扑鼻细膜微酶','用刀体第级卡住瓶口','级景区区南昆山泉水','制的名酒人沉醉','镀上清酒合名會社径正德年開始','造就瓶水果星球','个胶己人种的油相胶己人酿的好酒','在初九月牙柚总有说词','口就喝到颗梅子的生','南部接连高登的业平宁山脉东临静的业得里业海','来源自然福佳红玫瑰啤酒','日式梅酒千贺寿梅酒','醒酒器个红酒杯个海马刀把倒酒器酒塞套装','梅乃宿日式柚子果酒','世界遗产小粒摩香葡萄','托卡伊产区贵腐黄金产地','味和葡萄柚的余芳','比利时芙力荔枝啤酒','张裕贵晚采甜红葡萄酒','国盛外婆蜂蜜柚子酒','托卡伊萨座罗德尼甜葡萄酒','何为慕拉蓝钻冰酒','巴尔比莫斯卡托高泡甜白','小吻红茶利口酒','莫斯卡托甜型葡萄酒','调兑俏雅梅酒基调的时尚鸡尾酒','配料只有梅子糖基酒再无其他','甘长以酸枣皮味收尾','吉治利口酒水果冰淇淋','经典通化葡萄酒','帕拉丁意大利起泡酒','基酒原料精选自','束花莫斯卡托阿斯蒂低泡甜白低醇葡萄酒','梅果肉酸味甜味和用無過滤','以五粮液白酒为基酒','拉索尔菲甜红葡萄酒','山花之吻起泡白葡萄汁','七容优品柔和酱香型白酒','杨梅酒喝苏打水的比例为','木寒味木屑进入酒中造成软木塞污染','凯琦苏格兰单麦芽威士忌','蜜桃菠萝草莓和橙子花瓣的','意大利多来利起泡酒','蓝钻冰酒优选珍贵品种雷司令','建康果实轻柔片榨取汁','山山楂酒查酒香酒香酒','高干穗熟成梅酒','八木酒造的青短柚子酒','产品名称桃妖真酿蜜桃味起泡果酒','西柚汁俏雅梅酒=','红葡萄酒俏雅梅酒=','口梅子酒口脆青梅','年熟成微酸梅酒','白桃酒桃丁果肉冰块迷送香','阿苏始美托卡伊贵腐甜葡萄酒','户河内黑糖梅子味威士忌配制酒','芳歌完熟梅酒女士低度甜酒','落落饮酒小记开','所以品牌名称为狮子歌歌','牙利我深爱的托卡伊','根据尼尔森数据','程整年温度上下浮动低于是','要凌官方旗舰店','不同用适当的温度','古越龙山联合打造参与出品','那就幼定杯酒的时间吧','再口把他们吃掉','头扎红头绳眉眼儿像清明时节的柳叶天比天明媚自','上好的女儿红凝岁月神韵十八年隔世蕴藏天地精华','满陇尽是桂花雨','到际化高标准进行全新设计拥有','再用两级播起木塞','纤体六楼杯出道即位','思幕客贵腐系列','裁培与限造至今已有年的限','牡丹王丹晚干红','昆动酒杯让酒与空','厂微酶的悟静时刻','中中电中中便中中','用腔诚垫焕醒深藏宫廷','青梅为形李子为部','保莱金丝托卡伊','招财进禧微美梅','的小瓶装非常适合','感受柚子の酸甜','过度的酸涩感哦','口感营养怡到好处','适时不会因为糖分过多或','每口都仿佛咬在口中的柚子测',
            '容量刚好不浪费无需担心开瓶后喝不完','两种包装随机发','和建民有限责会','喝口狮子歌歌系枝清酒','波本桶陈年年以上','豆办农味力の口','度入门级日桃酒','即保证法定产区酒','连续五年新酒鉴评会',
            '放置时更加稳固增大表面摩擦力','德芙妮白起泡葡萄酒','德芙妮桃红起泡葡萄酒','远超食用酒精高度烈酒米酒等做基酒的果酒口感','饮用水糯米酒桂花汁≥等','唇点无醇起泡红葡萄汁','唇点无醇起泡葡萄汁',
            '新贝半甜白葡萄酒','里姆雪莱尔甜白起泡葡萄酒','橙香白桃乌龙酒','完熟梅酒度的酒精含量','玫瑰酿冰葡萄酒','双果味轻感萄萄酒','番荔枝椰奶味口','由雷司令黑皮诺丹菲特混酿而成','可保存个月但开封后的赏味佳期只有天',
            '【鸡尾酒做法】','用毛巾包住瓶口','白鹤梅子利口酒','三年陈绍兴花雕酒半干型','世界葡萄酒觉赛利口酒金骨','狮子歌歌红西柚果酒','户河内梅酒源头自户河内威士总原酒并辅以南高梅酿造','但蜜限酒龙其清鲜','宁治茶梅酒度数','贵冰帆蓝莓冰酒','柚香茶芳清列百','配以高于穗酒造本格烧耐酿造','托卡伊签奥苏贵腐甜葡萄酒',
            '国座梅使用大甘控兑の梅酒寸','兑俏雅梅酒基调的时','绍兴特产黄酒八年陈封坛花雕酒','甸牙利阿苏饸美要托卡伊贵腐甜葡萄酒','此款圈魔梅酒使用大颗粒的大分产莺宿梅','而法治通地县叶时培地都','相节办心丸办办忘','社美人の娘版记园',
            '人间下正经记欠酒手册','对它的香肤直器证不了','校若太阳升朝霞灼若芙','町了久术人米没取高梁川地下的伏','年托卡伊阿苏五等','爵香贵腐酒都入钻石般珍责','狮子歌歌各种鲜果酿就','荔枝酿酒新知杜明','②稻梓米酒的酿造技艺是通过代又代酿造师傅言传','山口造酒厂位于福网反留米市年间酿酒从未间断的酒','澳迪尼月夕玫瑰花酒','期工序外且在酿酒槽中自然','萌家增州杨梅酒','选自犯州产南高梅','中整酒造林式会社','上甄蒸酒六个轮次',
            '啤酒真露>>随意组合=啤烧酒','具格的江南花酒','喝口狮子歌歌荔枝清酒','便十葡萄根系的保温礼','鉴湖水称为酒之血把其酿造','正年威代尔冰酒在五女山','儿梅酒の饮物「完熟梅使','该柚子为装饰图片仅供参考',
            '青梅柔甚果花酒','贵妮冰枫放冰酒','狮子歌歌乌梅梅酒','馆娃宫桂花冬酿','办儿爵示列葡萄酒','梅酒酸造少毛年','万多金牌美莎米兰之花','越王【勾践】采用大夫【文种】谋','[五女山]框龙湖位于北纬度','口条和四种口站','[五女山]恒龙湖位于北纬度','引自日本殿堂南高梅',
            '顺来自自由和酒李白诗词分享','北野天满宫内有很多莺鸟栖身自山口家族庭院中开掘出','露嘉娜特雷比奥罗','不锈钢福中进行','国座南高梅使用','国隆南高梅使用',
            '金兹玛拉乌里集团','饮子方の寸寸防','夏子又子暑<冬以板寒典型的在盆地の氨候','选用日本歌山自桃','二年成酿只为响','[五女山恒龙湖位于北纬度','西岭雪山脚下青梅','休闲小豹几杯微刚刚好',
            '加拿大进口六支礼盒','贝瑞潮礼颜值实力兼真','香奈官方旗舰店','月到月间在贵腐菌发展程度深的葡萄园里','私大与注酒苏万物色乙','③补德超意孔过滤','年【国除大会】金算受算',
            '小红帽装瓶现场','住瓶口接着翘起木塞','山地文菲诺雪莉','【葡萄种类】雷司令','亲友馈赠礼真意切','亲友馈赠礼真意切','不同于劣查酒瓶','金黄色葡萄球离','意大利农业葡萄酒协会',
            '剔除坏掉的葡萄及葡萄','狮子歌歌马奶酒','日本の丨×のの果集酒','金酒俏雅梅酒=','瓶油柑酒半瓶都是油相汁','狮子歌歌马奶酒','日本の丨×のの果集酒','狮子歌歌杨枝甘露清酒','中酿酒葡萄遍地年以上老藤','度十红稀有臻酿','选白纪州产南高梅',
            '【东良只】梅乃宿酒造','苏州桥蜜桃米露','馆娃宫江南米酒','人雪兰山葡药酒业有限','安森受葡萄酒庄在带','大黑福选用油绳产黑糖','【梅酒冰淇淋】','提起两筑并按压按钮使',
            '地处山东省烟台市公司始建于','「上海诺德生物」','有时候必须承认','花小喵の桃酒评测','从平成年开始就任「负上清酒合名会社」的','份的山楂花之吻份','通明山荔枝酒喝下的每口','胶己人种的油相','消费者晋及牙利葡萄','晃杯观色深嘎闻香小口细品','精致毛查查的奶牛酒标','年历经年建造的张裕地下大酒窖完工','梅酒只喝南高梅','此情景的第五代酒庄主山口利七对天神使者所饮之神水','技艺称为酒之经络把其酿造的绍兴酒与育人等同对待其重视便可见斑','好山育好果好果酿好酒','及罗莎山脉与法国瑞士及意大利伦巴第利古里业','萌家靖州杨梅酒','世界上醉好的小甜小产区','特单阿尼山谷酒庄的历更起源于世纪在原有亚','历山大查查瓦泽王子的酒窖和限酒厂的基础上俄国','曾速年受赏全國新酒金膜',
            '再陶坛陈酿年最后经分离澄清','雷根特晚收半甜红','升水藏之干岁味常好','此情景的第五代酒庄主山口利七对大神使者所饮之神水','京料采用较好瓦伦西业西杆','储区应避免异味','①存储专业葡萄酒恒温恒湿',
            '滑向瓶肩避免出现浑浊','开瓶后的赏味期不宜超过天要尽早饮用','重金属较少含有多','像木塞避免因气压向题漏出','生理期的暖宝宝','是强烈结构是简','冷藏或加入冰坊','确保酒体始终保持',
            '地理的表示制度','产区对于级别贵腐','耐类物质质量高','裁培与限造至今已有年的限','人间下正经记酒手册','私九古过物古酒力','澳迪尼月夕蜜柚','良果中西部办第二阪东道路办通大阪办','阿苏始美签托卡伊','出湿波洛神华贵','硫影横斜暗香浮动胶胶河汉','町了久术人米没取高梁川地下的伏','参差红紫熟方好','波本桶陈年年以上','时光梅酒田江记酒片打造','怀化市洪江区珠家进情乡美酒厂出','年历经年建造的张裕地下大酒窖完工','技艺称为酒之经络把其酿造的绍兴酒与育人等同对待其重视便可见斑','及罗莎山脉与法国瑞士及意大利伦巴第利古里业','佐酒菜卤己安排','莫斯卡托白起泡','温哥华魔云起泡酒','相比要他的贵腐酒富盛酪自营','福酒才用金铜爷爷的原酒','狮子歌歌柚子梅酒','酒时浪野系果酒','年波本桶个月雪莉桶陈年','青梅柔甚果花酒','狮子歌歌荷叶梅酒','小正酒厂创业与年位于鹿儿岛跟葡萄酒的中心是波尔多威士忌的中心是','命之源水百利甜酒由蒸馅大','产自年以上老腰野树','梅酒酸造少毛年','白桃桑套玫瑰梅子','莱莉酒杨梅酒青梅酒桂花酒','液体黄金贵腐酒','吉治利口酒苏打水碎冰','日本原装进口老松酒造梅酒王','万通鲜果酒发酵型','德国柔丝伯爵巧克力甜红配制酒','青梅冰糖百花蜜酿造','莫斯卡托桃红起泡酒','包牙利托卡伊贵腐甜白葡萄酒','个以色列大柚子','卡内利地下大教堂','只使用与俏雅长期合作的梅农们','只使用与俏雅长期合作的梅农们精心','精选纪州南高梅','创牙利国际大赛金奖','根枝条上的同位置年才能再次结果','别小脆这瓶考基酒','酒师生涯作为目前较具盛名的','昔日的荒山秀岭','河武造株式會社径昭和時代開始使用','瓶怡到好处的微酶','质的关键是梅子与投入的时间成本','余家知名航空公司','越南波特嘉之夜','年月日苏格兰格拉斯量','来自自由和酒诗词分享','关于创牙利国家馆','罗维斯茶色波特','新限和制底业行限任公司','孩投桃予利口酒','金酒值雅检酒=身加入','奈良果中西部汇办第二阪奈道路办通大阪办','户河内梅子威十后','藏看限酒师的分','私大古注酒苏石物毛','私大古注酒苏石物办','静置存储定时间后萃取出大红袍西柚酒','加入特质发酵曲药发酵天后','以获得荔枝发酵酒','新限和制底业行限任公司','孩投桃予利口酒','金酒值雅检酒=身加入','奈良果中西部汇办第二阪奈道路办通大阪办','户河内梅子威十后','藏看限酒师的分','私大古注酒苏石物毛','私大古注酒苏石物办','静置存储定时间后萃取出大红袍西柚酒','加入特质发酵曲药发酵天后','以获得荔枝发酵酒','法定产区葡萄酒经历','在红葡萄酒中加入俏雅既能','莫斯卡托起泡酒容易入口','阿苏精华的残糖量为克升','使用当季糯米发酵后的酸和米查','吉治利口酒水果冰淇淋','长兴大责梅原汁与纯净伏特加碰撞而成','这款优质小贵腐残糖量高达',
            '梅酒发酵完毕后酒液和酒酸是在起的','梅果肉酸味甜味和用無過滤','佳酿葡萄酒被称为液体黄金','特加法国进口级白兰地','配以数年以上米酒窖藏','将青梅酿入桂花酒里','都是兄你而味道','年的发展有目共','已进驻盒马鲜生罗森酒类直供','底部凹槽使瓶体放置更平稳','食品应当有中文标签中文说明书标签说明书应当符合本法','登盘此是杨家果','月下独的四首其','私大古注酒色苏石物办','泛户初期宽文年年全知多共有','按照不同的规格进行','怡到好处的丹宁','班的时候小酚杯','全少编含看朵花瓣的精华','托卡伊贵腐系列','香芋倍加忠于冷暖紫知的本味','微酶快意如酿刚刚好','阿索罗超级普洛塞克级别','的首歌曲>给予酿酒','本味寒造家族新成员','东成年人调勿饮酒酒后请勿理驶','于品质好产量少般都用于本主供','五女山恒龙湖位于北纬度','红乙女酒造年担任社长的林田春野说我想打造不','人于年建立了思慕客酒庄直坚持家族','神明以析求酒造亲的繁樂杉藏元极具日本传','住瓶口接着翅起木型','澳迪尼月夕桃花酿露','私付物语仁毛酒水态石','【奈良票梅乃宿酒造','用第级翘起木塞','通化南面河地下大酒生','纹宫酒文创涤旨拾猫系列产品','国王行宫西拉桃红','高额值业女心桂花酒','选自犯州产南高柏','的搭配在梅酒的余味中可以感受茶单宁的微','怎么能少了泡泡的点级','州南高梅鲜果凳酵梅','江小果红动果酒','五斤杏酿造瓶杏酒','杏葡萄干蓝莓果酱以','艾露无醇起泡白葡萄汁','白酒蜂蜜冰糖种成分','冰红茶梅酒气泡水','热萃工艺提取葡萄汁','型甜型桃红葡萄酒','赤霞珠葡萄酿制','果汁气泡甜酒女士微熏果酒','杨桃橙汁姜汁蜂蜜小麦','三三二种随机发送瓶','三三种随机发送瓶','专业的防滑功能','鲜果精酿红西枯酒','籽露鲜酿玛瑙石榴酒','冰夹原浆葡萄酒','鲜果酵酿起泡酒','酒盛配托卡伊贵腐黄金产','鲜果精酸青耐酒','毫升果也柚子果酒','古风系列微藻时刻','必经的萃炼之路','洋洋酒酒地挥霍喜悦','爱女海莲娜拯救父业干危难','典雅仕文图点级','学习和教学始终充序看多纳托拉纳带的生活','时血清醒时血放肆','飞天仕女图图案彰','在拥有优良的团队同时交流看酸适工艺人们手工','的面孔冲饮上杯叙旧','新世界的螺旋帽','四月芳林何情情绿阴满地青梅小','新注息小心划手','南高梅赶走日疲意','兄弟国蜜欢聚时','轰肌音乐节助兴常备','纤体六楼杯出道即位','尼业加拉大澡布尼业加拉还孕','李鸿章签发营业准照光绪皇帝老师翁同解先生为张裕题写厂名','苏州桥与永辉超市盒马','蚕歌真迹与现代','长<克九名世界','慕拉庄园园主展肖华点评','出湿波洛神华贵','所以建议你们国国国拉','牛肉|海鲜类丁鸡肉了','带薪喝酒当不美盐','古朴傅就的外靓','日牛皮纸素色外盒','以及境内代理商的名称地址联系方式预包装食品没有中文标签中文说','≥果汁青梅酿造','西柚汁俏雅梅酒=','瓶毫升的青梅酒','口梅子酒口脆青梅','贵妮冰帆蓝莓冰酒','以清酒工艺标准酿制基酒','除了梅砂糖基酒','古法酿造传统黄酒为基酒','名杯莱尼诺查韵日葡萄起泡酒','白桃酒桃丁果肉冰块迷送香','智利葡菌酒产区','阿苏饸美等托卡伊贵腐甜葡萄酒所获奖项','点击查看价格>','桃妖开运招桃花','金黄配色飞天仕女','个品尝它们的人','年月中国商业企业管理协会','定制周转箱包装','更懂你的本格梅酒','人人都爱的度油相酒','零凌凌官方旗舰店','爱之湾喵喵礼盒装','大俄罗斯泰国日本和香港澳门等国家和地区还将继','提起新世界葡萄酒就不得不要说个国家','个月之所以选择晚采收是因为采','将成为甸牙利产品大型输出平台','全年高温日光张扬登夜温差大','对牙利的理解要从开始来自牙利','蜜拿薰衣草味汽泡酒','日本进口杂贺柚子酒女士果酒','施梅桂花酒采用当季的鲜桂花','酒庄旗舰系列的起泡酒','度的低度微酶梅肉的甜与梅汁','梅酒柚子酒组合','阿哈申尼半甜红葡萄酒','丰富的层次云岭冰酒的橡木桶陈年威代尔','种主要用于酿造葡萄酒的葡','知疆轻酸奶酒系列香草','这款装萝椰奶酒的配方','爱丝冰白葡萄酒','十果香和残存的橡木余味','酿入桂花陈年原酒','他们普遍关注表象','红覆盆子莓果实制作','通化红梅葡萄酒','柔丝伯爵贵族红葡萄配制酒','鲜榨荔枝汁天轻发酵','松次郎威士忌梅酒','贵妮冰机蓝莓冰酒','春村米酿桂花米酒','日本青梅酒梅子酒','超过种法定葡萄种植葡萄酒品类丰量','五女山干红葡萄酒支','五女山干红葡萄酒','辛迪娅西拉红葡萄酒','宝石红山葡萄酒','三品牌直营严谨品质','三百六十五小时','の笑颜私の周追求寸','订为此刻的愉悦碰材','漫薇思绪邀杯浅药动芳意','娜年十八可爱型','张裕葡小萄红酒','蜜黛甜自微汽泡酒',
            '托卡伊萨摩罗德尼甜白葡萄酒','单宁般是由葡萄籽皮及硬浸泡发酵而来或者是因为存','【梅酒莫吉托】','的山橙酒将山楂的营养成','足以应对瓶内压力','1件95折2件9折','密封结实确保产品能够更加安全到达您家','1281000','聚会小药微刚好','加拿大是当前全球受','画有一個傅统每年的6月10月12月都會在伊','要是再多份微醒',
            '泰起悲壮的离别之曲','保证品质始终如','是甜品的微点级','众所周知葡萄酒的保存要避光的','专业防滑底纹与瓶底凹槽设计','子味道与众不同其中的纪州南','参照','程序','评价','理由',
            '联保','此图','起算','参照','购物车','收货','故障','专卖店','仓发','图片展示','图中','理解','支持','参数','依据','洗涤剂洗衣粉','轻松洗净脏污','单脱漂脱','果脱水漂','店铺累计热销','注所标注',
            '踢脚板支架','需要安装','装修师傅','机器安装','安装区域','位置调整 ','水电环境','可拒绝支付','安装案例','安装附加费','等相关问题','水电安全','产品安装','选购指南','安心购','购买无忧','最高抽取','质保','价保权益','扫码','安装方式 ','大额神劵','观看直播领取惊喜','美的京东自营','店铺','价保','有限公司','安装条件','橱柜内壁','上海市','台面安装','范围内的偏差','免费测量','品牌资源','标准收费','新机器收到后','自来水管','说明书','开孔尺寸','实验试水','避免太阳','海尔洗碗机','安装位置及方式','出厂时','软管组件','数据来源','正常现象','进水管','排水管','排水管最大高度','春民室','浦东新区','银行账户中','试验方法','安装图','安装步骤','合标准要求','安装示意图','全部复制','样品符合','实验室书面','标准偏离','橱柜台面深度','厨具有限','长度宽度高度','专用章','见样品铭牌','排水管距离地面','符合标准子样品','外形尺寸','宽深高','报告编号','安装尺寸','项目符合标准','金数据的准确性','浦东新区民生路','委托方对样品','有限公司','报告编号','上海海关','浙江省','春民室','白条','承担相应','留言单号','少量余水','检测时残留','固定电机的装置','下方的底脚','放置时','工作日内发出','质量问题','型号对比','自有产品对比','检测报告'
           ]

# 产品页的网址 F12 + F5  打开网站得到 url= 'https://item.jd.com/100006287020.html'
headers_product = {
    'Cookie': '__jdv=122270672|direct|-|none|-|1683140809445; __jdu=1683140809445778268474; shshshfp=2d5cba76d442c52db3285828f7181c8b; shshshfpa=b2fb3533-5999-fc24-2778-6efc70124643-1683140809; shshshfpx=b2fb3533-5999-fc24-2778-6efc70124643-1683140809; rkv=1.0; areaId=2; ipLoc-djd=2-2830-51803-0; shshshfpb=mY67AKHokSnlK7jfcC2dA_Q; PCSYCityID=CN_310000_310100_0; 3AB9D23F7A4B3CSS=jdd034EQVFFJXJT67GU4QMOEUJVTELTJ75UQHRN5EVEFFITTAPKUQXB7BWWH4YSVRT5P6QYQBX2MF2SCXE6VS5DEDVMTOIEAAAAMH4VRPELIAAAAAC73OJHNZ5LEU5IX; _gia_d=1; xapieid=jdd034EQVFFJXJT67GU4QMOEUJVTELTJ75UQHRN5EVEFFITTAPKUQXB7BWWH4YSVRT5P6QYQBX2MF2SCXE6VS5DEDVMTOIEAAAAMH4VRPELIAAAAAC73OJHNZ5LEU5IX; jsavif=1; jsavif=1; shshshsID=ed18a64a1623046f03dbc6dd315d33fe_2_1683180688576; __jda=122270672.1683140809445778268474.1683140809.1683174522.1683180678.3; __jdb=122270672.2.1683140809445778268474|3.1683180678; __jdc=122270672; qrsc=3; 3AB9D23F7A4B3C9B=4EQVFFJXJT67GU4QMOEUJVTELTJ75UQHRN5EVEFFITTAPKUQXB7BWWH4YSVRT5P6QYQBX2MF2SCXE6VS5DEDVMTOIE',
    'Referer': 'https://www.jd.com/',
    'User-Agent':'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.87 Safari/537.36 SE 2.X MetaSr 1.0'
            }
# 评论 F12 + F5 url='https://club.jd.com/comment/productCommentSummaries.action?referenceIds=' + id
headers_Comment = {
    'Cookie': 'shshshfpa=0b807b04-4d4f-3a7f-d629-10f106f621a2-1633872638; shshshfpb=pAdmoXIM%2FwBxftmVodEIdjA%3D%3D; pinId=RaGwmgLCP48RoD9K0aiD6A; __jdu=700303314; pin=yasewang1980; unick=yasewang1980; _tp=lLX4wffka%2Bn0yNGR5f0sXA%3D%3D; _pst=yasewang1980; unpl=JF8EAK1nNSttUUtXAxlVGEYXTlgBW1lfQkQCPWFSAQkNSlFWSQdJEhB7XlVdXhRKEB9uYxRXX1NPUg4eBysSEXteXVdZDEsWC2tXVgQFDQ8VXURJQlZAFDNVCV9dSRZRZjJWBFtdT1xWSAYYRRMfDlAKDlhCR1FpMjVkXlh7VAQrBBoTEkpfVl5YOHsQM19XDVVcWkNdNRoyGiJSHwFSWFkOTxJOaWYEVlxaSVQAKwMrEQ; TrackID=1mA-3MChNxGfgi2tJelJEFhi4cLkalpT-77A-tHR0VJ6AkEAnD6mc3kxcU9AwqxW9a32uepULg8heNNT2rD77SsYiSVlVd8Ktk797zdL8NlA; ceshi3.com=000; __jdv=76161171|baidu-pinzhuan|t_288551095_baidupinzhuan|cpc|0f3d30c8dba7459bb52f2eb5eba8ac7d_0_81273f2e644d45f8b0c7fded04bc4c11|1654444674141; areaId=2; thor=B676E39D86228127F49AB34ECE98100788F79A99977CCA10D4B3010773DACDC3A6F686FAEDEBC6883606DC7C0BF3C3D678F78D677946D41FF6F64741D340EA1060B627488FBF62055A6A90979B08438FE4E5F43489C7E676C6BC5D279203BB8CFC5E6906ED388C2AEB76A50866C26D5D647DE3F1D41F9AB6A8AAFE0DB08BD74D9E98F5D9FF35917D060F8B915F3D5331; __jdc=122270672; __jda=122270672.700303314.1633872637.1653959646.1654444665.83; shshshfp=82d3109f28570d2d289cbbf26fac94bf; 3AB9D23F7A4B3C9B=E7PQA6YLEV6VUULZ5NSOFBRWSQAUMZ4BSH36TCRHXM4ODZE6TXCQZO3S7CCVCJK6THUCZPWHH54MAGYN7UZPWZXMPA; token=275a52ab2af14d693b58a6b3bb2661e2,3,919136; __tk=617898ebae08a907ee67788890ecb32f,3,919136; shshshsID=e26390f79ec94b8ebf3f2aeee3be4fe7_8_1654445437015; ip_cityCode=2817; ipLoc-djd=2-2817-51973-0; __jdb=122270672.11.700303314|83.1654444665',
    'Referer': 'https://item.jd.com/100024641402.html',
    'User-Agent':'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.87 Safari/537.36 SE 2.X MetaSr 1.0'
            }
#图片页的网址   url2='https://cd.jd.com/description/channel?skuId=10031491110391&mainSkuId=10021060592118&charset=utf-8&cdn=2'
headers_pic = {
    'Cookie': 'shshshfpb=mY67AKHokSnlK7jfcC2dA_Q; jsavif=1',
    'User-Agent':'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.87 Safari/537.36 SE 2.X MetaSr 1.0',
    'Host': 'cd.jd.com',
    'Referer': 'https://npcitem.jd.hk/'
            }


@st.cache_data
def get_pdt_detail(key_word):
    st.write(key_word,channel)
    aa = quote(key_word, safe=";/?:@&=+$,", encoding="utf-8")
    url='https://search.jd.com/Search?keyword=' + aa + '&enc=utf-8&wq=' + aa + '&pvid=c3db784d9601463e9e0414754546fb7b'

    url1=url
    response = requests.get(url, headers=headers_product)
    with open('dzdp.html',mode='w',encoding='utf-8') as f:                  #打开第一个产品的页面
        f.write(response.text) 
    with open ('dzdp.html', mode='r', encoding = 'utf-8') as f:
        dzdp1=f.read()
    html = etree.HTML(response.text) #把源文件再格式化，可以用xpath, 否则不能用

    pdt_urls=html.xpath('//div[starts-with(@class,"p-name p-name-type-2")]/a/@href')  #把页面里的所有产品的url down下来  
    pdt_url=['https:'+i for i in pdt_urls]

    url=pdt_url[0]                                                          #选取第一个产品 

    response = requests.get(url, headers=headers_product)
    with open('dzdp.html',mode='w',encoding='utf-8') as f:                  #打开第一个产品的页面
        f.write(response.text) 
    with open ('dzdp.html', mode='r', encoding = 'utf-8') as f:
        dzdp1=f.read()
    html = etree.HTML(response.text) 

    pic_url_desc='https:'+ re.findall("desc: \'(.*?)',",response.text)[0]          #获取图片页面的信息
    # print(pic_url_desc)

    pdt_info=html.xpath('//ul[starts-with(@class,"parameter2 p-parameter-list")]/li/text()') #获取产品描述信息
    pdt_info=(',').join(pdt_info)
    # # print(pdt_info)

    #处理图片页面的信息，拿到每个图片的地址，并下载
    response = requests.get(pic_url_desc,headers=headers_pic)              #获取图片页面
    with open('dzdp2.html',mode='w',encoding='utf-8') as f: 
        f.write(response.text) 

    with open ('dzdp2.html', mode='r', encoding = 'utf-8') as f:          #获取图片页面
        dzdp2=f.read()
    html = etree.HTML(response.text) #把源文件再格式化，可以用xpath, 否则不能用
    result=etree.tostring(html)

    df_pdt=pd.DataFrame({})
    pic_url=[]  #存储所有页面里的图片
    txts = []

    dzdp2=response.text
    headers = {'User-Agent':'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.87 Safari/537.36 SE 2.X MetaSr 1.0'}
    if dzdp2.count('lazyload'):
        pic_urls=re.findall("//img(.*?).jpg",dzdp2)  #         print('这个图片的地址是 layzyload 模式,地址是___:', product_url)
        for i in pic_urls:
            a='https://img'+i.replace('\\','')+'.jpg'
            pic_url.append(a)
    else:
        pic_urls=re.findall("//img(.*?)\); height",dzdp2) #         print('这个图片的地址是标准img模式,地址是___:', product_url)
        for i in pic_urls:
            a='https://img'+ i
            pic_url.append(a)#             print('这个图片的url地址是___:', a)

    pic_code=1
    # if os.path.exists(path2):
    kw_word=[]
    for i in pic_url:
        response=requests.get(i,headers =headers) 
        pic_name = key_word + '_' + str(pic_code) #key_wordEN 
        download=pic_name + '.jpg'                 #  path2 + '/'+ 
        with open(download, 'wb') as fd:
            for chunk in response.iter_content():
                fd.write(chunk)
        pic_code=pic_code+1
        result = ocr.ocr(download, cls=True)
        # st.write(result)

        temp=[]
        
        for line in result:
            for text in line:
                
                temp.append(text[1][0])
        # st.write(temp)
        txts.append(','.join(temp))       # 把一个图片的所有内容装到一个字符串里，方便key bert 解析去掉没有的信息
        
        aa= kw_model.extract_keywords((',').join(temp), 
                              keyphrase_ngram_range=(1,1), diversity=1, top_n=5) #use_mmr=True
        kw_word=kw_word + [j[0] for j in aa]
    # st.write(kw_word) # ['aaa','bbb','ccc'] 还需要再合并，才能装到df里去

    txts=(',').join(txts)
    pic_url=(',').join(pic_url)
    # df_pdt['brand']=pd.DataFrame([brand])
    df_pdt['key_word']=pd.DataFrame([key_word])    
    df_pdt['channel']=pd.DataFrame([channel])
    df_pdt['url1']=pd.DataFrame([url1])
    # df_pdt['share']=pd.DataFrame([share])
    df_pdt['price']=pd.DataFrame([0])
    df_pdt['txts']=pd.DataFrame([txts])
    df_pdt['pic_url_desc']=pd.DataFrame([pic_url_desc])
    df_pdt['pdt_info']=pd.DataFrame([pdt_info])
    df_pdt['pic_url']=pd.DataFrame([pic_url])
    df_pdt['kw_word']=pd.DataFrame([(',').join(kw_word)])

    return df_pdt,kw_word

###################################### main part ########################################################

st.set_page_config(page_title='Product Information Inquiry', page_icon=':bar_chart:', layout='wide')

st.write('-----------------------------------')
col1, col2 = st.columns([2,2])
with col1:
    title = st.text_input('产品型号或名称', '', key='具体产品型号，不要只输入品牌')
with col2:
    upload_models= st.file_uploader('请上传你需要的所有产品型号 xlsx格式')    
st.write('-----------------------------------')

st.write(title)

if len(title)>0:
    pdt_code=st.multiselect('请选择一个您要查询的产品:',([str(title)]))
else:
    models=pd.read_excel(upload_models)
    pdt_code=st.multiselect('请选择一个您要查询的产品:',
                   (models.models.values.tolist()),
                   (models.models.values.tolist()))

st.write(pdt_code)
categories=['Dishwash']  #'Laundry','Dryer'
channel='online'  #,'offline',

df_pdt_detail=pd.DataFrame({})

for key_word in pdt_code:
    temp=pd.DataFrame({})
    try:
        temp,kw_words = get_pdt_detail(key_word)
        df_pdt_detail=pd.concat([df_pdt_detail,temp])
    except Exception as e:
        continue

# st.dataframe(df_pdt_detail,use_container_width=True )

address=df_pdt_detail['url1'][0]
st.write(address)

col1, col2, col3, col4, col5, col6 = st.columns(6)
with col1:
    st.info('**型号**')
    st.caption(df_pdt_detail['key_word'][0])
with col2:
    st.info('**品牌**')
    st.caption('tbc')
with col3:
    st.info('**上市**')
    st.caption('tbc')
with col4:
    st.info('**份额**')
    st.caption('tbc')
with col5:
    st.info('**价格**')
    st.caption('tbc')
with col6:
    st.info('**网址**')
    st.write('**[@link](address)**')

st.caption(df_pdt_detail['pdt_info'][0])

st.write('-----------------------------------')

col1, col2, col3, col4, col5, col6, col7, col8, col9, col10, col11, col12 = st.columns(12, gap="small")
with col1:
    st.image(Image.open(str(pdt_code[0])+'_'+'1'+'.jpg')) #,width=320 ,use_column_width=True  
with col2:
    st.image(Image.open(str(pdt_code[0])+'_'+'2'+'.jpg'))
with col3:
    st.image(Image.open(str(pdt_code[0])+'_'+'3'+'.jpg'))
with col4:
    st.image(Image.open(str(pdt_code[0])+'_'+'4'+'.jpg'))
with col5:
    st.image(Image.open(str(pdt_code[0])+'_'+'5'+'.jpg'))
with col6:
    st.image(Image.open(str(pdt_code[0])+'_'+'6'+'.jpg'))
with col7:
    st.image(Image.open(str(pdt_code[0])+'_'+'7'+'.jpg')) #,width=320 ,use_column_width=True  
with col8:
    st.image(Image.open(str(pdt_code[0])+'_'+'8'+'.jpg'))
with col9:
    st.image(Image.open(str(pdt_code[0])+'_'+'9'+'.jpg'))
with col10:
    st.image(Image.open(str(pdt_code[0])+'_'+'10'+'.jpg'))
with col11:
    st.image(Image.open(str(pdt_code[0])+'_'+'11'+'.jpg'))
with col12:
    st.image(Image.open(str(pdt_code[0])+'_'+'12'+'.jpg'))


# st.write('-----主要卖点------------------------------')

df_kw=pd.DataFrame({})
df_kw['txts3']=pd.DataFrame(kw_words)

for i in dropword:
    df_kw['txts3']=df_kw['txts3'].apply(lambda x: str(x) if str(x).count(i)==0 else '')

df_kw['txts3']=df_kw['txts3'].apply(lambda x: str(x) if len(x)>4 else '')
df_kw.drop_duplicates(subset=['txts3'],inplace=True)
df_kw=df_kw[df_kw['txts3']!='']
df_kw.reset_index(drop=True,inplace=True)


kw_bert= kw_model.extract_keywords(','.join(df_kw['txts3'].values.tolist()), 
                            keyphrase_ngram_range=(8,8), diversity=0.9, top_n=16, use_mmr=True) #

kw_bert_final=pd.DataFrame({})
kw_bert_final['txts3'] = pd.DataFrame([j[0] for j in kw_bert])

st.dataframe(kw_bert_final,height=600,use_container_width=True)


# st.write('-----产品灵感------------------------------')

df_neg=pd.DataFrame({})
df_neg['Document']=pd.DataFrame(df_pdt_detail['txts'].values.tolist()[0].split(','))

for i in dropword:
    df_neg['Document']=df_neg['Document'].apply(lambda x: str(x) if str(x).count(i)==0 else '')

df_neg['Document']=df_neg['Document'].apply(lambda x: re.sub("[③②a-zA-Z0-9\s+\.\-\!\/_,$-%^*(+']+|[+——！℃=☆★×·，·。？、：；;《》“”~@#￥%……&*（）]+",'',str(x)))   # 去除标点及特殊符号
df_neg['Document']=df_neg['Document'].apply(lambda x: str(x) if len(x)>4 else '') #长度为1的字符
df_neg.drop_duplicates(subset=['Document'],inplace=True)
df_neg=df_neg[df_neg['Document']!='']
df_neg.reset_index(drop=True,inplace=True)

# st.write(df_neg)

model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
sentences = df_neg['Document'].values
sentence_embeddings = model.encode(sentences)

@st.cache_data
def cluster_cal(df_neg,num_clusters,sentence_embeddings): 
    # 使用 K-Means 对句子向量进行聚类
    from sklearn.cluster import AgglomerativeClustering
    cluster = AgglomerativeClustering(n_clusters=num_clusters, affinity='cosine', linkage='complete')
    clusters = cluster.fit_predict(sentence_embeddings)

    b=hyp.plot(sentence_embeddings,size=[6,6],fmt='-',ndims=3,n_clusters=num_clusters,reduce='UMAP',) 
    # print(b.xform_data[0].shape)

    vectors_x=b.xform_data[0][:,0]
    vectors_y=b.xform_data[0][:,1]    
    
    df_neg['clusters']=pd.DataFrame(clusters)
    df_neg['vectors_x']=pd.DataFrame(vectors_x)
    df_neg['vectors_y']=pd.DataFrame(vectors_y)

    cluster_num,labels=[],[]
    dfname=pd.DataFrame({})
    for i in range(num_clusters):
    #     print(i)
        if len(df_neg[df_neg['clusters']==i]['Document'].values.tolist())>6:
            cluster_num.append(i)
            label=(' | ').join(random.sample(df_neg[df_neg['clusters']==i]['Document'].values.tolist(),2))
            labels.append(label)
        else:
            cluster_num.append(i)
            label=(' | ').join(random.sample(df_neg[df_neg['clusters']==i]['Document'].values.tolist(),1))
            labels.append(label)
    dfname['clusters']=pd.DataFrame(cluster_num)
    dfname['labels']=pd.DataFrame(labels)
    df_neg=df_neg.merge(dfname,how='left',on='clusters')

    table=pd.pivot_table(df_neg,values=['vectors_x','vectors_y'],index='labels',aggfunc=np.mean) #,,columns=
   
    n=0
    table=pd.pivot_table(df_neg,values=['vectors_x','vectors_y'],index='labels',aggfunc=np.mean) #,,columns=

    import plotly_express as px  
    import plotly.graph_objects as go
    var=globals()
    
    var['fig'+str(n)] = px.scatter(table,x='vectors_x',y='vectors_y',size_max=40,text=table.index,#facet_col='color', ## ,marker = dict(color=color0)  color='color',
                    ) #,color_discrete_sequence=px.colors.qualitative.T10   color_discrete_map={"size_index>130":"Darkred",">80 $ <100":"Orange","<80":"Grey"} ,size='score'
    var['fig'+str(n)].update_layout(height=850,width=2000,template="plotly")#     plot_bgcolor='rgba(0,0,0,0)',
    var['fig'+str(n)].update_traces(textfont_size=12, textposition="top center", cliponaxis=False)  
    # # var['fig'+str(n)].update_yaxes(range=[-3000,4000])
    # st.plotly_chart(var['fig'+str(n)],use_container_width=True)

    return df_neg,var['fig'+str(n)]


num_clusters = int(st.number_input('聚类的数量'))
df_neg_final,fig=cluster_cal(df_neg,num_clusters,sentence_embeddings)

theme_plotly = None 


df_neg_final=df_neg_final[['clusters','Document','labels']]
col1, col2,col3 = st.columns([1.5,1.5,3],gap='small')
with col1:
    for i in range(0,int(num_clusters/2)):
        st.dataframe(df_neg_final[df_neg_final['clusters']==i],height=150,use_container_width=True)
with col2:
    for i in range(int(num_clusters/2),num_clusters):
        st.dataframe(df_neg_final[df_neg_final['clusters']==i],height=150,use_container_width=True)

with col3:
    st.plotly_chart(fig,use_container_width=True, width=900, height=1000, theme=theme_plotly)


# @st.cache_data
# def convert_df(df):
#     # IMPORTANT: Cache the conversion to prevent computation on every rerun
#     return df.to_csv().encode('utf_8_sig') 

# csv = convert_df(df_neg_final)

# st.download_button(
#     label="cluster file as CSV",
#     data=csv,
#     file_name='df_neg_final.csv',
#     mime='text/csv',
# )





# st.write('--------------Appendix---------------------')
# col1, col2, col3, col4, col5 = st.columns(5, gap="small")
# with col1:
#     st.dataframe(df_kw.iloc[:20,:],use_container_width=True) #height=500, 
# with col2:
#     st.dataframe(df_kw.iloc[20:40,:],use_container_width=True)
# with col3:
#    st.dataframe(df_kw.iloc[30:60,:],use_container_width=True)
# with col4:
#     st.dataframe(df_kw.iloc[60:80,:],use_container_width=True)
# with col5:
#     st.dataframe(df_kw.iloc[80:,:],use_container_width=True)

# st.write('-----------------------------------')

