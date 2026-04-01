//@ aux-crate: zerocopy=unnecessary-transmute-path-remap-ice-140277-trans.rs
//@ check-pass
// tests for a regression in linting for unnecessary transmutes
// where a span was inacessible for snippet procuring,
// when remap-path-prefix was set, causing a panic.

fn bytes_at_home(x: [u8; 4]) -> u32 {
    unsafe { zerocopy::transmute!(x) }
}
fn main() {}
