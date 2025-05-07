//@ known-bug: #138089
#![feature(generic_const_exprs)]
#![feature(min_generic_const_args)]
#![feature(inherent_associated_types)]
struct OnDiskDirEntry<'a> {}

impl<'a> OnDiskDirEntry<'a> {
    const LFN_FRAGMENT_LEN: i64 = 2;

    fn lfn_contents() -> [char; Self::LFN_FRAGMENT_LEN] {
        loop {}
    }
}
