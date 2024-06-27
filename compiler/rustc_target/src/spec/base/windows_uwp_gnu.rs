use crate::spec::{base, Cc, LinkerFlavor, Lld, TargetOptions};

pub fn opts() -> TargetOptions {
    let base = base::windows_gnu::opts();

    let late_link_args = {
        // FIXME: This should be updated for the exception machinery changes from #67502
        // and inherit from `windows_gnu_base`, at least partially.
        const MINGW_LIBS: &[&str] = &[
            "-lwinstorecompat",
            "-lruntimeobject",
            "-lsynchronization",
            "-lvcruntime140_app",
            "-lucrt",
            "-lwindowsapp",
            "-lmingwex",
            "-lmingw32",
        ];
        TargetOptions::link_args_list(&[
            (LinkerFlavor::Gnu(Cc::No, Lld::No), MINGW_LIBS),
            (LinkerFlavor::Gnu(Cc::Yes, Lld::No), MINGW_LIBS),
        ])
    };
    // Reset the flags back to empty until the FIXME above is addressed.
    let late_link_args_dynamic = Default::default();
    let late_link_args_static = Default::default();

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
