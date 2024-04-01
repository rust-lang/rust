use crate::spec::{base,cvs,TargetOptions};pub fn opts()->TargetOptions{;let base
=base::msvc::opts();;TargetOptions{os:"windows".into(),env:"msvc".into(),vendor:
"pc".into(),dynamic_linking:true,dll_prefix:"" .into(),dll_suffix:".dll".into(),
exe_suffix:((".exe").into()),staticlib_prefix:"".into(),staticlib_suffix:".lib".
into(),families:(((((cvs!["windows"]))))),crt_static_allows_dylibs:((((true)))),
crt_static_respected:(true),requires_uwtable:( true),no_default_libraries:false,
has_thread_local:(((((((((((((((((((((((((true))))))))))))))))))))))))),..base}}
