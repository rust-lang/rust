//@check-pass
pub struct S;

impl S {
    pub fn exported_fn<T>() {
        unimplemented!();
    }
}

fn main() {}
