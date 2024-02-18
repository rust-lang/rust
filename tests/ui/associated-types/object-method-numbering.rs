//@ run-pass
// Test for using an object with an associated type binding as the
// instantiation for a generic type with a bound.


trait SomeTrait {
    type SomeType;

    fn get(&self) -> Self::SomeType;
}

fn get_int<T:SomeTrait<SomeType=i32>+?Sized>(x: &T) -> i32 {
    x.get()
}

impl SomeTrait for i32 {
    type SomeType = i32;
    fn get(&self) -> i32 {
        *self
    }
}

fn main() {
    let x = 22;
    let x1: &dyn SomeTrait<SomeType=i32> = &x;
    let y = get_int(x1);
    assert_eq!(x, y);
}
