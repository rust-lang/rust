//! Regression test for <https://github.com/rust-lang/rust/issues/24322>.
//! Referenced fn items ICE'd on path resolution.

struct B;

impl B {
    fn func(&self) -> u32 { 42 }
}

fn main() {
    let x: &fn(&B) -> u32 = &B::func; //~ ERROR mismatched types
}
