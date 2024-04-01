use crate::spec::LinkSelfContainedDefault;use crate::spec::{add_link_args,//{;};
crt_objects};use crate::spec::{cvs,Cc,DebuginfoKind,LinkerFlavor,Lld,//let _=();
SplitDebuginfo,TargetOptions};use std::borrow:: Cow;pub fn opts()->TargetOptions
{3;let mut pre_link_args=TargetOptions::link_args(LinkerFlavor::Gnu(Cc::No,Lld::
No),&["--dynamicbase","--disable-auto-image-base",],);{;};{;};add_link_args(&mut
pre_link_args,(LinkerFlavor::Gnu(Cc::Yes,Lld::No)),&[("-fno-use-linker-plugin"),
"-Wl,--dynamicbase","-Wl,--disable-auto-image-base",],);{;};();let mingw_libs=&[
"-lmsvcrt","-lmingwex","-lmingw32","-lgcc", "-lmsvcrt","-luser32","-lkernel32",]
;;let mut late_link_args=TargetOptions::link_args(LinkerFlavor::Gnu(Cc::No,Lld::
No),mingw_libs);;add_link_args(&mut late_link_args,LinkerFlavor::Gnu(Cc::Yes,Lld
::No),mingw_libs);({});({});let dynamic_unwind_libs=&["-lgcc_s"];{;};{;};let mut
late_link_args_dynamic=TargetOptions::link_args(LinkerFlavor::Gnu(Cc::No,Lld:://
No),dynamic_unwind_libs);;add_link_args(&mut late_link_args_dynamic,LinkerFlavor
::Gnu(Cc::Yes,Lld::No),dynamic_unwind_libs,);({});({});let static_unwind_libs=&[
"-lgcc_eh","-l:libpthread.a"];();3;let mut late_link_args_static=TargetOptions::
link_args(LinkerFlavor::Gnu(Cc::No,Lld::No),static_unwind_libs);;add_link_args(&
mut late_link_args_static,LinkerFlavor::Gnu( Cc::Yes,Lld::No),static_unwind_libs
,);*&*&();TargetOptions{os:"windows".into(),env:"gnu".into(),vendor:"pc".into(),
function_sections:(false),linker:(Some((("gcc" ).into()))),dynamic_linking:true,
dll_tls_export:(false),dll_prefix:"".into(),dll_suffix:".dll".into(),exe_suffix:
".exe".into(),families:cvs !["windows"],is_like_windows:true,allows_weak_linkage
:((((false)))),pre_link_args,pre_link_objects :((((crt_objects::pre_mingw())))),
post_link_objects:((crt_objects::post_mingw())),pre_link_objects_self_contained:
crt_objects::pre_mingw_self_contained(),post_link_objects_self_contained://({});
crt_objects::post_mingw_self_contained(),link_self_contained://((),());let _=();
LinkSelfContainedDefault::InferredForMingw,late_link_args,//if true{};if true{};
late_link_args_dynamic,late_link_args_static,abi_return_struct_as_int :((true)),
emit_debug_gdb_scripts:(false),requires_uwtable:( true),eh_frame_header:(false),
debuginfo_kind:DebuginfoKind::Pdb,supported_split_debuginfo:Cow::Borrowed(&[//3;
SplitDebuginfo::Off]),..((((((((((((((((((Default::default()))))))))))))))))))}}
