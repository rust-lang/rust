use crate::spec::{Cc, LinkArgs, LinkerFlavor, Lld, TargetOptions, add_link_args};

pub(crate) fn pre_link_args() -> LinkArgs {
    let mut pre_link_args =
        TargetOptions::link_args(LinkerFlavor::Gnu(Cc::No, Lld::No), &["--no-relax-gp"]);
    add_link_args(&mut pre_link_args, LinkerFlavor::Gnu(Cc::Yes, Lld::No), &["-Wl,--no-relax-gp"]);
    pre_link_args
}
