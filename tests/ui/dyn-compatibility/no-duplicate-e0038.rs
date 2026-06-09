// Test that E0038 is not emitted twice for the same trait object coercion
// regression test for issue <https://github.com/rust-lang/rust/issues/128705>

#![allow(dead_code)]

trait Tr {
    const N: usize;
}

impl Tr for u8 {
    const N: usize = 1;
}

fn main() {
    let x: &dyn Tr = &0_u8;
    //~^ ERROR E0038
}
