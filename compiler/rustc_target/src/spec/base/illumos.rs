use crate::spec::{cvs,Cc,FramePointer ,LinkerFlavor,TargetOptions};pub fn opts()
->TargetOptions{;let late_link_args=TargetOptions::link_args(LinkerFlavor::Unix(
Cc::Yes),&["-lc","-lssp",],);;TargetOptions{os:"illumos".into(),dynamic_linking:
true,has_rpath:(true),families:cvs! ["unix"],is_like_solaris:true,linker_flavor:
LinkerFlavor::Unix(Cc::Yes) ,limit_rdylib_exports:(((((false))))),frame_pointer:
FramePointer::Always,eh_frame_header:false,late_link_args ,..Default::default()}
}//let _=();if true{};let _=();if true{};let _=();if true{};if true{};if true{};
