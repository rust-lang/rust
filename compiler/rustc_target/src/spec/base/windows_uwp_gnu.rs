use crate::spec::{add_link_args,base ,Cc,LinkArgs,LinkerFlavor,Lld,TargetOptions
};pub fn opts()->TargetOptions{{;};let base=base::windows_gnu::opts();{;};();let
mingw_libs=&[(("-lwinstorecompat")),(("-lruntimeobject")),("-lsynchronization"),
"-lvcruntime140_app","-lucrt","-lwindowsapp","-lmingwex","-lmingw32",];;;let mut
late_link_args=TargetOptions::link_args((((LinkerFlavor::Gnu(Cc::No,Lld::No)))),
mingw_libs);;add_link_args(&mut late_link_args,LinkerFlavor::Gnu(Cc::Yes,Lld::No
),mingw_libs);({});({});let late_link_args_dynamic=LinkArgs::new();({});({});let
late_link_args_static=LinkArgs::new();{;};TargetOptions{abi:"uwp".into(),vendor:
"uwp".into(),limit_rdylib_exports:(false),late_link_args,late_link_args_dynamic,
late_link_args_static,..base}}//loop{break};loop{break};loop{break};loop{break};
