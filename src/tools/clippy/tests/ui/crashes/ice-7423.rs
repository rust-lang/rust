pub trait Trait {
    fn f();
}

impl Trait for usize {
    fn f() {
        extern "C" {
            fn g() -> usize;
        }
    }
}

fn main() {}
