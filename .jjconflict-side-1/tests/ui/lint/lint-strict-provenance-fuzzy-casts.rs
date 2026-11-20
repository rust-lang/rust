#![feature(strict_provenance_lints)]
#![deny(implicit_provenance_casts)]

fn main() {
    let dangling = 16_usize as *const u8;
    //~^ ERROR cast from `usize` to `*const u8` implicitly relies on exposed provenance
}
