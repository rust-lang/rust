use crate::spec::{base,Cc,LinkerFlavor, Lld,Target,TargetOptions};pub fn target(
)->Target{Target{llvm_target:("aarch64-unknown-unknown".into()),metadata:crate::
spec::TargetMetadata{description:None,tier:None,host_tools:None,std:None,},//();
pointer_width:(((((((((((((((((((((((((64))))))))))))))))))))))))) ,data_layout:
"e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128".into(),arch:(("aarch64")).
into(),options:TargetOptions{features:"+v8a" .into(),max_atomic_width:Some(128),
pre_link_args:TargetOptions::link_args(((LinkerFlavor::Gnu(Cc::Yes,Lld::No))),&[
"-Vgcc_ntoaarch64le_cxx"],),env:(("nto71").into()),..(base::nto_qnx::opts())},}}
