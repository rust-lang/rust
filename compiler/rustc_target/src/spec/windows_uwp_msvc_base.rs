use crate::spec::{LinkerFlavor, TargetOptions};

pub fn opts() -> TargetOptions {
    let mut opts = super::windows_msvc_base::opts();

    opts.abi = "uwp".into();
    opts.vendor = "uwp".into();
    opts.add_pre_link_args(LinkerFlavor::Msvc, &["/APPCONTAINER", "mincore.lib"]);

    opts
}
