//@ compile-flags:-C debuginfo=1

pub trait TraitWithDefaultMethod : Sized {
    fn method(self) {
        ()
    }
}

struct MyStruct;

impl TraitWithDefaultMethod for MyStruct { }

pub fn main() {
    MyStruct.method();
}
