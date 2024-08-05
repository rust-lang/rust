#![deny(dead_code)]

trait Trait {
    const UNUSED_CONST: i32; //~ ERROR associated constant `UNUSED_CONST` is never used
    const USED_CONST: i32;

    fn foo(&self) {}
}

pub struct T(());

impl Trait for T {
    const UNUSED_CONST: i32 = 0;
    const USED_CONST: i32 = 1;
}

fn main() {
    T(()).foo();
    T::USED_CONST;
}
