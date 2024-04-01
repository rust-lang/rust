use crate::spec::{base,cvs,FramePointer, Target,TargetOptions};pub fn target()->
Target{Target{llvm_target:("thumbv5te-none-eabi". into()),metadata:crate::spec::
TargetMetadata{description:None,tier:None,host_tools:None,std:None,},//let _=();
pointer_width:((((((32)))))),arch:(((((((((("arm" ))))).into()))))),data_layout:
"e-m:e-p:32:32-Fi8-i64:64-v128:64:128-a:0:32-n32-S64".into(),options://let _=();
TargetOptions{abi:((((((("eabi"))).into())))),asm_args:cvs!["-mthumb-interwork",
"-march=armv5te","-mlittle-endian",],features://((),());((),());((),());((),());
"+soft-float,+strict-align,+atomics-32".into(),frame_pointer:FramePointer:://();
MayOmit,main_needs_argc_argv:false,atomic_cas :false,has_thumb_interworking:true
,..(((((((((((((((((((((((((((base::thumb::opts())))))))))))))))))))))))))))},}}
