// Regression test for https://github.com/rust-lang/rust/issues/52060
// The compiler shouldn't ICE in this case
static A: &'static [u32] = &[1];
static B: [u32; 1] = [0; A.len()];
//~^ ERROR [E0013]

fn main() {}
