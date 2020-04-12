use crate::spec::{LinkerFlavor, LldFlavor, TargetOptions};

pub fn opts() -> TargetOptions {
    let mut opts = super::windows_msvc_base::opts();

    let new_link_args = vec!["/APPCONTAINER".to_string()];
    opts.link_args
        .get_mut(&LinkerFlavor::Msvc)
        .unwrap()
        .unordered_right_overridable
        .extend(new_link_args.clone());
    opts.link_args
        .get_mut(&LinkerFlavor::Lld(LldFlavor::Link))
        .unwrap()
        .unordered_right_overridable
        .extend(new_link_args);

    let pre_link_args_msvc = vec!["mincore.lib".to_string()];
    opts.pre_link_args.get_mut(&LinkerFlavor::Msvc).unwrap().extend(pre_link_args_msvc.clone());
    opts.pre_link_args
        .get_mut(&LinkerFlavor::Lld(LldFlavor::Link))
        .unwrap()
        .extend(pre_link_args_msvc);

    opts
}
