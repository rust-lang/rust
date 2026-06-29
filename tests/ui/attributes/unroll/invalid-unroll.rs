#![feature(loop_hints)]
#![crate_type = "lib"]

pub fn main() {
    #[unroll(please)] //~ ERROR malformed `unroll` attribute input
    for _ in 0..10 {}

    #[unroll("never")] //~ ERROR malformed `unroll` attribute input
    for _ in 0..10 {}

    #[unroll()] //~ ERROR malformed `unroll` attribute input
    for _ in 0..10 {}

    #[unroll(-1)] //~ ERROR expected a literal
    for _ in 0..10 {}

    #[unroll(1.5)] //~ ERROR malformed `unroll` attribute input
    for _ in 0..10 {}
}
