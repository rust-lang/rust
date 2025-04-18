//@ check-pass

pub trait Trait {
    fn f();
}

impl Trait for usize {
    fn f() {
        unsafe extern "C" {
            fn g() -> usize;
        }
    }
}

fn main() {}
