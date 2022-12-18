use crate::spec::{LinkerFlavor, Lld, TargetOptions};

pub fn opts() -> TargetOptions {
    let mut opts = super::windows_msvc_base::opts();

    opts.abi = "uwp".into();
    opts.vendor = "uwp".into();
    opts.add_pre_link_args(LinkerFlavor::Msvc(Lld::No), &["/APPCONTAINER", "mincore.lib"]);

    opts
}
