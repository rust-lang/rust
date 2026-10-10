// Test for issue #157568
//@ compile-flags: -Zpolonius
//@ check-pass

#![warn(rust_2024_compatibility)]
pub struct F;
impl std::fmt::Debug for F {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("F").finish_non_exhaustive()
    }
}

fn main() {}
