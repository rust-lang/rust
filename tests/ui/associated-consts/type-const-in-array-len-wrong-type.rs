#![feature(generic_const_exprs)]
//~^ WARN the feature `generic_const_exprs` is incomplete
#![feature(min_generic_const_args)]
//~^ WARN the feature `min_generic_const_args` is incomplete
#![feature(inherent_associated_types)]
//~^ WARN the feature `inherent_associated_types` is incomplete

struct OnDiskDirEntry<'a>(&'a ());

impl<'a> OnDiskDirEntry<'a> {
    type const LFN_FRAGMENT_LEN: i64 = 2;

    fn lfn_contents() -> [char; Self::LFN_FRAGMENT_LEN] {
        //~^ ERROR the constant `2` is not of type `usize`
        loop {}
    }
}

fn main() {}
