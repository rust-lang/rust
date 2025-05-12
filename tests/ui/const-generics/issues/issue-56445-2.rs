// Regression test for https://github.com/rust-lang/rust/issues/56445#issuecomment-502095133
struct OnDiskDirEntry<'a> { _s: &'a usize }

impl<'a> OnDiskDirEntry<'a> {
    const LFN_FRAGMENT_LEN: usize = 2;

    fn lfn_contents(&self) -> [char; Self::LFN_FRAGMENT_LEN] { loop { } }
    //~^ ERROR: generic `Self` types are currently not permitted in anonymous constants
}

fn main() {}
