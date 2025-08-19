// Regression test for https://github.com/rust-lang/rust/issues/52060
// The compiler shouldn't ICE in this case

static mut A: &'static [u32] = &[1];
static B: [u32; 1] = [0; unsafe { A.len() }];
//~^ ERROR: mutable global memory

fn main() {}
