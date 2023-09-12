use crate::spec::{base::windows_msvc, LinkerFlavor, Lld, TargetOptions};

pub fn opts() -> TargetOptions {
    let mut opts = windows_msvc::opts();

    opts.abi = "uwp".into();
    opts.vendor = "uwp".into();
    opts.add_pre_link_args(LinkerFlavor::Msvc(Lld::No), &["/APPCONTAINER", "mincore.lib"]);

    opts
}
