use crate::spec::{LinkerFlavor, Lld, TargetOptions, base};

pub(crate) fn opts() -> TargetOptions {
    let mut opts =
        TargetOptions { abi: "uwp".into(), vendor: "uwp".into(), ..base::windows_msvc::opts() };

    opts.add_pre_link_args(LinkerFlavor::Msvc(Lld::No), &["/APPCONTAINER", "mincore.lib"]);

    opts
}
