use crate::spec::{Abi, LinkerFlavor, Lld, TargetOptions, Vendor, base};

pub(crate) fn opts() -> TargetOptions {
    let mut opts = base::windows_msvc::opts();

    opts.abi = Abi::Uwp;
    opts.vendor = Vendor::Uwp;
    opts.add_pre_link_args(LinkerFlavor::Msvc(Lld::No), &["/APPCONTAINER", "mincore.lib"]);

    opts
}
