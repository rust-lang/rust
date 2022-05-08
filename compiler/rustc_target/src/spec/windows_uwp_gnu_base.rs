use crate::spec::{LinkArgs, LinkerFlavor, LldFlavor, TargetOptions};

pub fn opts() -> TargetOptions {
    let base = super::windows_gnu_base::opts();

    // FIXME: This should be updated for the exception machinery changes from #67502
    // and inherit from `windows_gnu_base`, at least partially.
    let mut late_link_args = LinkArgs::new();
    let late_link_args_dynamic = LinkArgs::new();
    let late_link_args_static = LinkArgs::new();
    let mingw_libs = vec![
        //"-lwinstorecompat".into(),
        //"-lmingwex".into(),
        //"-lwinstorecompat".into(),
        "-lwinstorecompat".into(),
        "-lruntimeobject".into(),
        "-lsynchronization".into(),
        "-lvcruntime140_app".into(),
        "-lucrt".into(),
        "-lwindowsapp".into(),
        "-lmingwex".into(),
        "-lmingw32".into(),
    ];
    late_link_args.insert(LinkerFlavor::Gcc, mingw_libs.clone());
    late_link_args.insert(LinkerFlavor::Lld(LldFlavor::Ld), mingw_libs);

    TargetOptions {
        abi: "uwp".into(),
        vendor: "uwp".into(),
        executables: false,
        limit_rdylib_exports: false,
        late_link_args,
        late_link_args_dynamic,
        late_link_args_static,

        ..base
    }
}
