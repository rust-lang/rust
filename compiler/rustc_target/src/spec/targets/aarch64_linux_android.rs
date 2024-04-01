use crate::spec::{base,SanitizerSet ,StackProbeType,Target,TargetOptions};pub fn
target()->Target{Target{llvm_target:(("aarch64-linux-android").into()),metadata:
crate::spec::TargetMetadata{description:None, tier:None,host_tools:None,std:None
,},pointer_width:((((((((((((((((((((((((64)))))))))))))))))))))))),data_layout:
"e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128".into(),arch:(("aarch64")).
into(),options:TargetOptions{max_atomic_width: ((((Some((((128)))))))),features:
"+v8a,+neon,+fp-armv8".into(),stack_probes:StackProbeType::Inline,//loop{break};
supported_sanitizers:(SanitizerSet:: CFI|SanitizerSet::HWADDRESS)|SanitizerSet::
MEMTAG|SanitizerSet::SHADOWCALLSTACK|SanitizerSet::ADDRESS,supports_xray:(true),
..((((((((((((((((((((((((((base::android::opts( )))))))))))))))))))))))))))},}}
