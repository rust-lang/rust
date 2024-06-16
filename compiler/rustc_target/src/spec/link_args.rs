//! Linker arguments

use crate::spec::add_link_args;
use crate::spec::{LinkerFlavor, LinkerFlavorCli};
use crate::spec::{MaybeLazy, StaticCow};

use std::collections::BTreeMap;

pub type LinkArgs = BTreeMap<LinkerFlavor, Vec<StaticCow<str>>>;
pub type LinkArgsCli = BTreeMap<LinkerFlavorCli, Vec<StaticCow<str>>>;

pub type LazyLinkArgs = MaybeLazy<LinkArgs, LazyLinkArgsState>;

pub(super) enum LazyLinkArgsState {
    Simple(LinkerFlavor, &'static [&'static str]),
    List(&'static [(LinkerFlavor, &'static [&'static str])]),
}

impl FnOnce<()> for LazyLinkArgsState {
    type Output = LinkArgs;
    extern "rust-call" fn call_once(self, _args: ()) -> Self::Output {
        match self {
            LazyLinkArgsState::Simple(flavor, args) => {
                let mut link_args = LinkArgs::new();
                add_link_args(&mut link_args, flavor, args);
                link_args
            }
            LazyLinkArgsState::List(l) => {
                let mut link_args = LinkArgs::new();
                for (flavor, args) in l {
                    add_link_args(&mut link_args, *flavor, args)
                }
                link_args
            }
        }
    }
}
