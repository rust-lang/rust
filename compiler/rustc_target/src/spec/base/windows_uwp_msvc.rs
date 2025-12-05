use crate::spec::{Abi, LinkerFlavor, Lld, TargetOptions, base};

pub(crate) fn opts() -> TargetOptions {
    let mut opts =
        TargetOptions { abi: Abi::Uwp, vendor: "uwp".into(), ..base::windows_msvc::opts() };

    opts.add_pre_link_args(LinkerFlavor::Msvc(Lld::No), &["/APPCONTAINER", "mincore.lib"]);

    opts
}
