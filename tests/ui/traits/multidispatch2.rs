//@ run-pass

use std::fmt::Debug;

trait MyTrait<T> {
    fn get(&self) -> T;
}

impl<T> MyTrait<T> for T
    where T : Default
{
    fn get(&self) -> T {
        Default::default()
    }
}

#[derive(Copy, Clone)]
struct MyType {
    dummy: usize
}

impl MyTrait<usize> for MyType {
    fn get(&self) -> usize { self.dummy }
}

fn test_eq<T,M>(m: M, v: T)
where T : Eq + Debug,
      M : MyTrait<T>
{
    assert_eq!(m.get(), v);
}

pub fn main() {
    test_eq(22_usize, 0_usize);

    let value = MyType { dummy: 256 + 22 };
    test_eq(value, value.dummy);
}
