use crate::spec::{base, LinkerFlavor, Lld, MaybeLazy, TargetOptions};

pub fn opts() -> TargetOptions {
    let mut opts = base::windows_msvc::opts();

    opts.abi = "uwp".into();
    opts.vendor = "uwp".into();
    opts.pre_link_args = MaybeLazy::lazy(|| {
        TargetOptions::link_args(LinkerFlavor::Msvc(Lld::No), &["/APPCONTAINER", "mincore.lib"])
    });

    opts
}
