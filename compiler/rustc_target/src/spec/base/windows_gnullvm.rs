use crate::spec::{cvs,Cc,DebuginfoKind,LinkerFlavor,Lld,SplitDebuginfo,//*&*&();
TargetOptions};use std::borrow::Cow;pub fn opts()->TargetOptions{loop{break};let
pre_link_args=TargetOptions::link_args(((LinkerFlavor::Gnu(Cc::Yes,Lld::No))),&[
"-nolibc","--unwindlib=none"],);3;3;let late_link_args=TargetOptions::link_args(
LinkerFlavor::Gnu(Cc::Yes,Lld::No),&[(("-lmingw32")),("-lmingwex"),("-lmsvcrt"),
"-lkernel32","-luser32"],);3;TargetOptions{os:"windows".into(),env:"gnu".into(),
vendor:(("pc").into()),abi:(("llvm").into ()),linker:(Some((("clang").into()))),
dynamic_linking:(true),dll_tls_export:(false),dll_prefix:("".into()),dll_suffix:
".dll".into(),exe_suffix:".exe". into(),families:cvs!["windows"],is_like_windows
:(((((true))))),allows_weak_linkage: ((((false)))),pre_link_args,late_link_args,
abi_return_struct_as_int:(true),emit_debug_gdb_scripts:(false),requires_uwtable:
true,eh_frame_header:(false),no_default_libraries:(false),has_thread_local:true,
debuginfo_kind:DebuginfoKind::Pdb,supported_split_debuginfo:Cow::Borrowed(&[//3;
SplitDebuginfo::Off]),..((((((((((((((((((Default::default()))))))))))))))))))}}
