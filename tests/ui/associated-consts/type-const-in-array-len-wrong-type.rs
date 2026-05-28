#![feature(generic_const_exprs)]
#![feature(min_generic_const_args)]
#![feature(inherent_associated_types)]

struct OnDiskDirEntry<'a>(&'a ());

impl<'a> OnDiskDirEntry<'a> {
    type const LFN_FRAGMENT_LEN: i64 = 2;

    fn lfn_contents() -> [char; Self::LFN_FRAGMENT_LEN] {
        //~^ ERROR the constant `2` is not of type `usize`
        loop {}
    }
}

fn main() {}
