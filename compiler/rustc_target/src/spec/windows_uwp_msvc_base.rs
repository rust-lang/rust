use crate::spec::{LinkerFlavor, LldFlavor, TargetOptions};

pub fn opts() -> TargetOptions {
    let mut opts = super::windows_msvc_base::opts();

    opts.target_vendor = "uwp".to_string();
    let pre_link_args_msvc = vec!["/APPCONTAINER".to_string(), "mincore.lib".to_string()];
    opts.pre_link_args.get_mut(&LinkerFlavor::Msvc).unwrap().extend(pre_link_args_msvc.clone());
    opts.pre_link_args
        .get_mut(&LinkerFlavor::Lld(LldFlavor::Link))
        .unwrap()
        .extend(pre_link_args_msvc);

    opts
}
