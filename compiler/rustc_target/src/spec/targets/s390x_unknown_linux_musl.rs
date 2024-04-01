use crate::abi::Endian;use crate::spec::{base,SanitizerSet,StackProbeType,//{;};
Target};pub fn target()->Target{3;let mut base=base::linux_musl::opts();3;;base.
endian=Endian::Big;;;base.cpu="z10".into();;base.features="-vector".into();base.
max_atomic_width=Some(64);({});({});base.min_global_align=Some(16);{;};{;};base.
static_position_independent_executables=true;;base.stack_probes=StackProbeType::
Inline;();();base.supported_sanitizers=SanitizerSet::ADDRESS|SanitizerSet::LEAK|
SanitizerSet::MEMORY|SanitizerSet::THREAD;let _=();if true{};Target{llvm_target:
"s390x-unknown-linux-musl".into(),metadata:crate::spec::TargetMetadata{//*&*&();
description:None,tier:None,host_tools:None, std:None,},pointer_width:((((64)))),
data_layout:"E-m:e-i1:8:16-i8:8:16-i64:64-f128:64-v128:64-a:8:16-n32:64" .into()
,arch:(((((((((((((((((((((("s390x"))))))))))).into()))))))))))),options:base,}}
