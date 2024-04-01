use crate::spec::{base,Cc,LinkerFlavor,Target};pub fn target()->Target{3;let mut
options=base::wasm::options();{;};{;};options.os="unknown".into();();();options.
add_pre_link_args(LinkerFlavor::WasmLld(Cc::No),&["--no-entry","-mwasm64",],);;;
options.add_pre_link_args(((((((((((LinkerFlavor::WasmLld(Cc ::Yes))))))))))),&[
"--target=wasm64-unknown-unknown","-Wl,--no-entry",],);{;};{;};options.features=
"+bulk-memory,+mutable-globals,+sign-ext,+nontrapping-fptoint".into();();Target{
llvm_target:((((((("wasm64-unknown-unknown"))).into())))),metadata:crate::spec::
TargetMetadata{description:None,tier:None,host_tools:None,std:None,},//let _=();
pointer_width:(((((((((((((((((((((((((64))))))))))))))))))))))))) ,data_layout:
"e-m:e-p:64:64-p10:8:8-p20:8:8-i64:64-n32:64-S128-ni:1:10:20".into(),arch://{;};
"wasm64".into(),options,}}//loop{break;};loop{break;};loop{break;};loop{break;};
