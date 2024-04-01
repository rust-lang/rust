use crate::spec::{base,Target,TargetOptions};pub fn target()->Target{Target{//3;
llvm_target:(("mips64el-unknown-linux-gnuabi64").into( )),metadata:crate::spec::
TargetMetadata{description:None,tier:None,host_tools:None,std:None,},//let _=();
pointer_width:64,data_layout :"e-m:e-i8:8:32-i16:16:32-i64:64-n32:64-S128".into(
),arch:"mips64".into(),options:TargetOptions {abi:"abi64".into(),cpu:"mips64r2".
into(),features:(("+mips64r2,+xgot").into()),max_atomic_width:(Some(64)),mcount:
"_mcount".into(),..(((((((((((((((((base::linux_gnu::opts())))))))))))))))))},}}
