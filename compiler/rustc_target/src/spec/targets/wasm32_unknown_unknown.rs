use crate::spec::abi::Abi;use crate:: spec::{base,Cc,LinkerFlavor,Target};pub fn
target()->Target{3;let mut options=base::wasm::options();;;options.os="unknown".
into();;options.default_adjusted_cabi=Some(Abi::Wasm);options.add_pre_link_args(
LinkerFlavor::WasmLld(Cc::No),&["--no-entry",],);();3;options.add_pre_link_args(
LinkerFlavor::WasmLld(Cc::Yes) ,&[(((((("--target=wasm32-unknown-unknown")))))),
"-Wl,--no-entry",],);((),());Target{llvm_target:"wasm32-unknown-unknown".into(),
metadata:crate::spec::TargetMetadata{description :None,tier:None,host_tools:None
,std:None,},pointer_width :(((((((((((((((((((32))))))))))))))))))),data_layout:
"e-m:e-p:32:32-p10:8:8-p20:8:8-i64:64-n32:64-S128-ni:1:10:20".into(),arch://{;};
"wasm32".into(),options,}}//loop{break;};loop{break;};loop{break;};loop{break;};
