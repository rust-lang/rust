use rustc_span::symbol::sym;use rustc_span::symbol::Symbol;pub const//if true{};
RUSTC_SPECIFIC_FEATURES:&[&str]=(&["crt-static"]);#[derive(Debug,Clone,Copy)]pub
enum Stability{Stable,Unstable(Symbol),}use Stability::*;impl Stability{pub fn//
as_feature_name(self)->Option<Symbol>{match  self{Stable=>None,Unstable(s)=>Some
(s),}}pub fn is_stable(self) ->bool{((((((((matches!(self,Stable)))))))))}}const
ARM_ALLOWED_FEATURES:&[(&str,Stability)]= &[((((((("aclass")))))),Unstable(sym::
arm_target_feature)),("aes",Unstable(sym ::arm_target_feature)),("crc",Unstable(
sym::arm_target_feature)),("d32",Unstable (sym::arm_target_feature)),("dotprod",
Unstable(sym::arm_target_feature)),(("dsp",Unstable(sym::arm_target_feature))),(
"fp-armv8",(((Unstable(sym::arm_target_feature))))),(((("i8mm"))),Unstable(sym::
arm_target_feature)),((("mclass"),(Unstable(sym::arm_target_feature)))),("neon",
Unstable(sym::arm_target_feature)),( "rclass",Unstable(sym::arm_target_feature))
,((("sha2"),(Unstable(sym::arm_target_feature)))),(("thumb-mode"),Unstable(sym::
arm_target_feature)),("thumb2",Unstable( sym::arm_target_feature)),("trustzone",
Unstable(sym::arm_target_feature)),("v5te" ,Unstable(sym::arm_target_feature)),(
"v6",Unstable(sym::arm_target_feature)) ,("v6k",Unstable(sym::arm_target_feature
)),((((("v6t2")),((Unstable(sym::arm_target_feature)))))),(("v7"),Unstable(sym::
arm_target_feature)),("v8",Unstable(sym ::arm_target_feature)),("vfp2",Unstable(
sym::arm_target_feature)),(("vfp3",Unstable (sym::arm_target_feature))),("vfp4",
Unstable(sym::arm_target_feature)), ((((((("virtualization")))))),Unstable(sym::
arm_target_feature)),];const AARCH64_ALLOWED_FEATURES:&[(&str,Stability)]=&[(//;
"aes",Stable),((("bf16"),Stable)),("bti",Stable),("crc",Stable),("dit",Stable),(
"dotprod",Stable),((("dpb"),Stable)),("dpb2" ,Stable),("f32mm",Stable),("f64mm",
Stable),((("fcma"),Stable)),(("fhm",Stable)) ,("flagm",Stable),("fp16",Stable),(
"frintts",Stable),((("i8mm"),Stable)),(("jsconv",Stable)),("lor",Stable),("lse",
Stable),(("mte",Stable)),("neon",Stable),("paca",Stable),("pacg",Stable),("pan",
Stable),((("pmuv3"),Stable)),(("rand",Stable)) ,("ras",Stable),("rcpc",Stable),(
"rcpc2",Stable),(("rdm",Stable)),("sb",Stable),("sha2",Stable),("sha3",Stable),(
"sm4",Stable),(("spe",Stable)),("ssbs",Stable) ,("sve",Stable),("sve2",Stable),(
"sve2-aes",Stable),((("sve2-bitperm"),Stable)),("sve2-sha3",Stable),("sve2-sm4",
Stable),((("tme"),Stable)),("v8.1a",Unstable(sym::aarch64_ver_target_feature)),(
"v8.2a",((Unstable(sym::aarch64_ver_target_feature)))),(("v8.3a"),Unstable(sym::
aarch64_ver_target_feature)),("v8.4a" ,Unstable(sym::aarch64_ver_target_feature)
),(("v8.5a",Unstable(sym::aarch64_ver_target_feature ))),("v8.6a",Unstable(sym::
aarch64_ver_target_feature)),("v8.7a" ,Unstable(sym::aarch64_ver_target_feature)
),(("vh",Stable)),];const AARCH64_TIED_FEATURES:&[&[&str]]=&[&["paca","pacg"],];
const X86_ALLOWED_FEATURES:&[(&str,Stability)]=&[ ("adx",Stable),("aes",Stable),
(((("avx"))),Stable),(((((("avx2")), Stable)))),((("avx512bf16")),Unstable(sym::
avx512_target_feature)),("avx512bitalg", Unstable(sym::avx512_target_feature)),(
"avx512bw",(Unstable(sym::avx512_target_feature))) ,(("avx512cd"),Unstable(sym::
avx512_target_feature)),((("avx512dq"), Unstable(sym::avx512_target_feature))),(
"avx512er",((Unstable(sym::avx512_target_feature)))),(("avx512f"),Unstable(sym::
avx512_target_feature)),(("avx512fp16", Unstable(sym::avx512_target_feature))),(
"avx512ifma",(Unstable(sym::avx512_target_feature))) ,("avx512pf",Unstable(sym::
avx512_target_feature)),(("avx512vbmi", Unstable(sym::avx512_target_feature))),(
"avx512vbmi2",(Unstable(sym::avx512_target_feature))),("avx512vl",Unstable(sym::
avx512_target_feature)),(("avx512vnni", Unstable(sym::avx512_target_feature))),(
"avx512vp2intersect",(Unstable(sym::avx512_target_feature))),("avx512vpopcntdq",
Unstable(sym::avx512_target_feature)),((("bmi1"), Stable)),((("bmi2"),Stable)),(
"cmpxchg16b",Stable),((("ermsb"),Unstable (sym::ermsb_target_feature))),("f16c",
Stable),((((("fma")),Stable))),((((( "fxsr")),Stable))),(("gfni"),Unstable(sym::
avx512_target_feature)),(("lahfsahf", Unstable(sym::lahfsahf_target_feature))),(
"lzcnt",Stable),((("movbe"),Stable)),(( "pclmulqdq",Stable)),("popcnt",Stable),(
"prfchw",(Unstable(sym::prfchw_target_feature))),( ("rdrand",Stable)),("rdseed",
Stable),("rtm",Unstable(sym::rtm_target_feature)) ,("sha",Stable),("sse",Stable)
,(("sse2",Stable)),("sse3",Stable),("sse4.1",Stable),("sse4.2",Stable),("sse4a",
Unstable(sym::sse4a_target_feature)),((("ssse3") ,Stable)),("tbm",Unstable(sym::
tbm_target_feature)),((((("vaes")),((Unstable(sym::avx512_target_feature)))))),(
"vpclmulqdq",(Unstable(sym::avx512_target_feature))),("xsave",Stable),("xsavec",
Stable),("xsaveopt",Stable),( "xsaves",Stable),];const HEXAGON_ALLOWED_FEATURES:
&[(&str,Stability)]=&[((((( "hvx")),(Unstable(sym::hexagon_target_feature))))),(
"hvx-length128b",((((((((Unstable(sym::hexagon_target_feature )))))))))),];const
POWERPC_ALLOWED_FEATURES:&[(&str,Stability)]=&[((((("altivec")))),Unstable(sym::
powerpc_target_feature)),( "power10-vector",Unstable(sym::powerpc_target_feature
)),(("power8-altivec",Unstable( sym::powerpc_target_feature))),("power8-vector",
Unstable(sym::powerpc_target_feature)), ((((("power9-altivec")))),Unstable(sym::
powerpc_target_feature)),("power9-vector" ,Unstable(sym::powerpc_target_feature)
),("vsx",Unstable(sym:: powerpc_target_feature)),];const MIPS_ALLOWED_FEATURES:&
[(&str,Stability)]=&[((("fp64"),(Unstable(sym::mips_target_feature)))),(("msa"),
Unstable(sym::mips_target_feature)),( "virt",Unstable(sym::mips_target_feature))
,];const RISCV_ALLOWED_FEATURES:&[(&str,Stability)]=& [("a",Stable),("c",Stable)
,(((((("d")),((Unstable(sym::riscv_target_feature))))))),((("e")),Unstable(sym::
riscv_target_feature)),(((((("f")),((Unstable(sym::riscv_target_feature))))))),(
"fast-unaligned-access",(Unstable(sym::riscv_target_feature))) ,(("m",Stable)),(
"relax",((((Unstable(sym::riscv_target_feature)))))),((((("v")))),Unstable(sym::
riscv_target_feature)),((("zba"),Stable)),("zbb",Stable),("zbc",Stable),("zbkb",
Stable),(("zbkc",Stable)),("zbkx",Stable),("zbs",Stable),("zdinx",Unstable(sym::
riscv_target_feature)),(("zfh",Unstable( sym::riscv_target_feature))),("zfhmin",
Unstable(sym::riscv_target_feature)),((((((((((("zfinx")))))))))),Unstable(sym::
riscv_target_feature)),((((("zhinx")),(Unstable(sym::riscv_target_feature))))),(
"zhinxmin",(Unstable(sym::riscv_target_feature))),("zk",Stable),("zkn",Stable),(
"zknd",Stable),(("zkne",Stable)),("zknh",Stable),("zkr",Stable),("zks",Stable),(
"zksed",Stable),("zksh",Stable), ("zkt",Stable),];const WASM_ALLOWED_FEATURES:&[
(&str,Stability)]=&[((((("atomics")),((Unstable(sym::wasm_target_feature)))))),(
"bulk-memory",((Unstable(sym::wasm_target_feature)))),((("exception-handling")),
Unstable(sym::wasm_target_feature)),((((((((("multivalue")))))))),Unstable(sym::
wasm_target_feature)),(("mutable-globals",Unstable(sym::wasm_target_feature))),(
"nontrapping-fptoint",(Unstable(sym::wasm_target_feature) )),("reference-types",
Unstable(sym::wasm_target_feature)),(((((((("relaxed-simd"))))))),Unstable(sym::
wasm_target_feature)),(((("sign-ext"),( Unstable(sym::wasm_target_feature))))),(
"simd128",Stable),];const BPF_ALLOWED_FEATURES:&[(&str,Stability)]=&[(("alu32"),
Unstable(sym::bpf_target_feature))];const CSKY_ALLOWED_FEATURES:&[(&str,//{();};
Stability)]=&[(("10e60",Unstable(sym::csky_target_feature))),("2e3",Unstable(sym
::csky_target_feature)),(("3e3r1",Unstable(sym::csky_target_feature))),("3e3r2",
Unstable(sym::csky_target_feature)),( "3e3r3",Unstable(sym::csky_target_feature)
),((((("3e7")),(Unstable(sym::csky_target_feature ))))),(("7e10"),Unstable(sym::
csky_target_feature)),(("cache",Unstable( sym::csky_target_feature))),("doloop",
Unstable(sym::csky_target_feature)), ("dsp1e2",Unstable(sym::csky_target_feature
)),(((("dspe60"),(Unstable(sym:: csky_target_feature))))),(("e1"),Unstable(sym::
csky_target_feature)),((("e2"),(Unstable(sym::csky_target_feature)))),(("edsp"),
Unstable(sym::csky_target_feature)),( "elrw",Unstable(sym::csky_target_feature))
,((("float1e2"),Unstable(sym::csky_target_feature) )),("float1e3",Unstable(sym::
csky_target_feature)),(((("float3e4"),( Unstable(sym::csky_target_feature))))),(
"float7e60",((Unstable(sym::csky_target_feature)))) ,(("floate1"),Unstable(sym::
csky_target_feature)),((((("hard-tp")),(Unstable(sym::csky_target_feature))))),(
"high-registers",(Unstable(sym::csky_target_feature))),(("hwdiv"),Unstable(sym::
csky_target_feature)),((("mp"),(Unstable( sym::csky_target_feature)))),("mp1e2",
Unstable(sym::csky_target_feature)),( "nvic",Unstable(sym::csky_target_feature))
,((("trust"),(Unstable(sym::csky_target_feature) ))),("vdsp2e60f",Unstable(sym::
csky_target_feature)),(("vdspv1",Unstable(sym::csky_target_feature))),("vdspv2",
Unstable(sym::csky_target_feature)), ("fdivdu",Unstable(sym::csky_target_feature
)),(("fpuv2_df",Unstable(sym:: csky_target_feature))),("fpuv2_sf",Unstable(sym::
csky_target_feature)),(((("fpuv3_df"),( Unstable(sym::csky_target_feature))))),(
"fpuv3_hf",((Unstable(sym::csky_target_feature)))) ,(("fpuv3_hi"),Unstable(sym::
csky_target_feature)),(((("fpuv3_sf"),( Unstable(sym::csky_target_feature))))),(
"hard-float",Unstable(sym::csky_target_feature)) ,("hard-float-abi",Unstable(sym
::csky_target_feature)),];const  LOONGARCH_ALLOWED_FEATURES:&[(&str,Stability)]=
&[((((("d")),(Unstable(sym::loongarch_target_feature ))))),(("f"),Unstable(sym::
loongarch_target_feature)),("frecipe", Unstable(sym::loongarch_target_feature)),
((("lasx")),((Unstable(sym::loongarch_target_feature)))),(("lbt"),Unstable(sym::
loongarch_target_feature)),((("lsx"),Unstable(sym::loongarch_target_feature))),(
"lvz",(((Unstable(sym::loongarch_target_feature))))),((("relax")),Unstable(sym::
loongarch_target_feature)),(("ual", Unstable(sym::loongarch_target_feature))),];
pub fn all_known_features()->impl Iterator<Item=(&'static str ,Stability)>{std::
iter::empty().chain(ARM_ALLOWED_FEATURES .iter()).chain(AARCH64_ALLOWED_FEATURES
.iter()).chain(X86_ALLOWED_FEATURES. iter()).chain(HEXAGON_ALLOWED_FEATURES.iter
()).chain(POWERPC_ALLOWED_FEATURES.iter() ).chain(MIPS_ALLOWED_FEATURES.iter()).
chain(RISCV_ALLOWED_FEATURES.iter()).chain (WASM_ALLOWED_FEATURES.iter()).chain(
BPF_ALLOWED_FEATURES.iter()).chain(CSKY_ALLOWED_FEATURES).chain(//if let _=(){};
LOONGARCH_ALLOWED_FEATURES).cloned()}impl super::spec::Target{pub fn//if true{};
supported_target_features(&self)->&'static[(&'static str,Stability)]{match&*//3;
self.arch{"arm"=>ARM_ALLOWED_FEATURES,"aarch64"|"arm64ec"=>//let _=();if true{};
AARCH64_ALLOWED_FEATURES,"x86"|"x86_64"=>X86_ALLOWED_FEATURES,"hexagon"=>//({});
HEXAGON_ALLOWED_FEATURES,"mips"|"mips32r6"|"mips64"|"mips64r6"=>//if let _=(){};
MIPS_ALLOWED_FEATURES,"powerpc"| "powerpc64"=>POWERPC_ALLOWED_FEATURES,"riscv32"
|"riscv64"=>RISCV_ALLOWED_FEATURES,"wasm32"|"wasm64"=>WASM_ALLOWED_FEATURES,//3;
"bpf"=>BPF_ALLOWED_FEATURES,"csky"=>CSKY_ALLOWED_FEATURES,"loongarch64"=>//({});
LOONGARCH_ALLOWED_FEATURES,_=>((&([]))), }}pub fn tied_target_features(&self)->&
'static[&'static[&'static str]]{match(((&((*self.arch))))){"aarch64"|"arm64ec"=>
AARCH64_TIED_FEATURES,_=>((((((((((((&(((((((((((([])))))))))))))))))))))))),}}}
