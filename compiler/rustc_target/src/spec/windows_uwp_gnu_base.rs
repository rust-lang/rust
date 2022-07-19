use crate::spec::{LinkArgs, LinkerFlavor, TargetOptions};

pub fn opts() -> TargetOptions {
    let base = super::windows_gnu_base::opts();

    // FIXME: This should be updated for the exception machinery changes from #67502
    // and inherit from `windows_gnu_base`, at least partially.
    let mingw_libs = &[
        "-lwinstorecompat",
        "-lruntimeobject",
        "-lsynchronization",
        "-lvcruntime140_app",
        "-lucrt",
        "-lwindowsapp",
        "-lmingwex",
        "-lmingw32",
    ];
    let mut late_link_args = TargetOptions::link_args(LinkerFlavor::Ld, mingw_libs);
    super::add_link_args(&mut late_link_args, LinkerFlavor::Gcc, mingw_libs);
    // Reset the flags back to empty until the FIXME above is addressed.
    let late_link_args_dynamic = LinkArgs::new();
    let late_link_args_static = LinkArgs::new();

    TargetOptions {
        abi: "uwp".into(),
        vendor: "uwp".into(),
        limit_rdylib_exports: false,
        late_link_args,
        late_link_args_dynamic,
        late_link_args_static,

        ..base
    }
}
