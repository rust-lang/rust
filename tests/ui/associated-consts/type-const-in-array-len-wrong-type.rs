#![feature(
    generic_const_exprs,
    gca_min_const_items,
    gca_macroless_args,
    inherent_associated_types
)]

use std::gca;

struct OnDiskDirEntry<'a>(&'a ());

impl<'a> OnDiskDirEntry<'a> {
    const LFN_FRAGMENT_LEN: i64 = gca!(2);

    fn lfn_contents() -> [char; Self::LFN_FRAGMENT_LEN] {
        //~^ ERROR the constant `2` is not of type `usize`
        loop {}
    }
}

fn main() {}
