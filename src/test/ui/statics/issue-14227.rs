// revisions: mir thir
// [thir]compile-flags: -Z thir-unsafeck

extern "C" {
    pub static symbol: u32;
}
static CRASH: u32 = symbol;
//~^ ERROR use of extern static is unsafe and requires

fn main() {}
