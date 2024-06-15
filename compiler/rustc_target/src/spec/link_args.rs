//! Linker arguments

use crate::spec::{LinkerFlavor, LinkerFlavorCli};
use crate::spec::{MaybeLazy, TargetOptions};
use crate::spec::StaticCow;

use std::collections::BTreeMap;

pub type LinkArgs = BTreeMap<LinkerFlavor, Vec<StaticCow<str>>>;
pub type LinkArgsCli = BTreeMap<LinkerFlavorCli, Vec<StaticCow<str>>>;

pub type LazyLinkArgs = MaybeLazy<LinkArgs, LazyLinkArgsState>;

pub(super) enum LazyLinkArgsState {
    Simple(LinkerFlavor, &'static [&'static str])
}

impl FnOnce<()> for LazyLinkArgsState {
    type Output = LinkArgs;
    extern "rust-call" fn call_once(self, _args: ()) -> Self::Output {
        match self {
            LazyLinkArgsState::Simple(flavor, args) =>
                TargetOptions::link_args_base(flavor, args),
        }
    }
}
