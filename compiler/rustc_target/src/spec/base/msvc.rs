use crate::spec::{DebuginfoKind, LinkerFlavor,Lld,SplitDebuginfo,TargetOptions};
use std::borrow::Cow;pub fn opts()->TargetOptions{loop{break};let pre_link_args=
TargetOptions::link_args(LinkerFlavor::Msvc(Lld::No),&["/NOLOGO"]);loop{break;};
TargetOptions{linker_flavor:(LinkerFlavor::Msvc(Lld ::No)),dll_tls_export:false,
is_like_windows:(true),is_like_msvc:true,pre_link_args,abi_return_struct_as_int:
true,emit_debug_gdb_scripts:(((false ))),split_debuginfo:SplitDebuginfo::Packed,
supported_split_debuginfo:((Cow::Borrowed(((&(( [SplitDebuginfo::Packed]))))))),
debuginfo_kind:DebuginfoKind::Pdb,..((((((((((((Default::default()))))))))))))}}
