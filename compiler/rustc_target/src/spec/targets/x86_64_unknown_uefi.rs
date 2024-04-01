use crate::{abi::call::Conv,spec::{base,Target},};pub fn target()->Target{();let
mut base=base::uefi_msvc::opts();;;base.cpu="x86-64".into();base.plt_by_default=
false;;;base.max_atomic_width=Some(64);;;base.entry_abi=Conv::X86_64Win64;;base.
features="-mmx,-sse,+soft-float".into();if true{};let _=||();Target{llvm_target:
"x86_64-unknown-windows".into(),metadata:crate::spec::TargetMetadata{//let _=();
description:None,tier:None,host_tools:None, std:None,},pointer_width:((((64)))),
data_layout://((),());((),());((),());let _=();((),());((),());((),());let _=();
"e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
.into(),arch:(((((((((((((((((("x86_64"))))))))).into()))))))))),options:base,}}
