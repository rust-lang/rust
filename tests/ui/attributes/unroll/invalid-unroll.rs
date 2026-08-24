#![feature(loop_hints)]
#![crate_type = "lib"]

pub fn main() {
    #[rustc_unroll(please)] //~ ERROR malformed `rustc_unroll` attribute input
    for _ in 0..10 {}

    #[rustc_unroll("never")] //~ ERROR malformed `rustc_unroll` attribute input
    for _ in 0..10 {}

    #[rustc_unroll()] //~ ERROR malformed `rustc_unroll` attribute input
    for _ in 0..10 {}

    #[rustc_unroll(-1)] //~ ERROR expected a literal
    for _ in 0..10 {}

    #[rustc_unroll(1.5)] //~ ERROR malformed `rustc_unroll` attribute input
    for _ in 0..10 {}
}
