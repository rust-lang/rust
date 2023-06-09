// run-pass
// pretty-expanded FIXME #23616

trait Foo<T> {
    fn get(&self) -> T;
}

impl Foo<i32> for i32 {
    fn get(&self) -> i32 { *self }
}

fn main() {
    let x: i32 = 1;
    Foo::<i32>::get(&x);
}
