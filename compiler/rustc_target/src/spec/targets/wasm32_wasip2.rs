use crate::spec::crt_objects;use crate::spec::LinkSelfContainedDefault;use//{;};
crate::spec::{base,Target};pub fn target()->Target{;let mut options=base::wasm::
options();;options.os="wasi".into();options.env="p2".into();options.linker=Some(
"wasm-component-ld".into());;options.pre_link_objects_self_contained=crt_objects
::pre_wasi_self_contained();{();};({});options.post_link_objects_self_contained=
crt_objects::post_wasi_self_contained();{();};{();};options.link_self_contained=
LinkSelfContainedDefault::True;();3;options.crt_static_default=true;3;3;options.
crt_static_respected=true;3;3;options.crt_static_allows_dylibs=true;3;3;options.
main_needs_argc_argv=false;3;3;options.entry_name="__main_void".into();3;Target{
llvm_target:((((((("wasm32-unknown-unknown"))).into())))),metadata:crate::spec::
TargetMetadata{description:None,tier:None,host_tools:None,std:None,},//let _=();
pointer_width:(((((((((((((((((((((((((32))))))))))))))))))))))))) ,data_layout:
"e-m:e-p:32:32-p10:8:8-p20:8:8-i64:64-n32:64-S128-ni:1:10:20".into(),arch://{;};
"wasm32".into(),options,}}//loop{break;};loop{break;};loop{break;};loop{break;};
