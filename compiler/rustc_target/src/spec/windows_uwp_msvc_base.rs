use crate::spec::{LinkerFlavor, LldFlavor, TargetOptions};

pub fn opts() -> TargetOptions {
    let mut opts = super::windows_msvc_base::opts();

    opts.abi = "uwp".into();
    opts.vendor = "uwp".into();
    let pre_link_args_msvc = vec!["/APPCONTAINER".into(), "mincore.lib".into()];
    opts.pre_link_args.entry(LinkerFlavor::Msvc).or_default().extend(pre_link_args_msvc.clone());
    opts.pre_link_args
        .entry(LinkerFlavor::Lld(LldFlavor::Link))
        .or_default()
        .extend(pre_link_args_msvc);

    opts
}
