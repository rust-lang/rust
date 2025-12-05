#![feature(strict_provenance_lints)]
#![deny(fuzzy_provenance_casts)]

fn main() {
    let dangling = 16_usize as *const u8;
    //~^ ERROR strict provenance disallows casting integer `usize` to pointer `*const u8`
}
