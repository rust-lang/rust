//@ check-pass

pub struct A {
    pub x: u32,
}

#[deny(unused_comparisons)]
pub fn foo(y: u32) -> A {
    A {
        #[allow(unused_comparisons)]
        x: if y < 0 { 1 } else { 2 },
    }
}

fn main() {}
