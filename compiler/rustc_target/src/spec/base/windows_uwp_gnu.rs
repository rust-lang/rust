use crate::spec::{add_link_args, base, Cc, LinkArgs, LinkerFlavor, Lld, MaybeLazy, TargetOptions};

pub fn opts() -> TargetOptions {
    let base = base::windows_gnu::opts();

    let late_link_args = MaybeLazy::lazy(|| {
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
        let mut late_link_args =
            TargetOptions::link_args_base(LinkerFlavor::Gnu(Cc::No, Lld::No), mingw_libs);
        add_link_args(&mut late_link_args, LinkerFlavor::Gnu(Cc::Yes, Lld::No), mingw_libs);
        late_link_args
    });
    // Reset the flags back to empty until the FIXME above is addressed.
    let late_link_args_dynamic = MaybeLazy::lazy(LinkArgs::new);
    let late_link_args_static = MaybeLazy::lazy(LinkArgs::new);

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
