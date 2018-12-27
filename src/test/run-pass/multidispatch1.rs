use std::fmt::Debug;

trait MyTrait<T> {
    fn get(&self) -> T;
}

#[derive(Copy, Clone)]
struct MyType {
    dummy: usize
}

impl MyTrait<usize> for MyType {
    fn get(&self) -> usize { self.dummy }
}

impl MyTrait<u8> for MyType {
    fn get(&self) -> u8 { self.dummy as u8 }
}

fn test_eq<T,M>(m: M, v: T)
where T : Eq + Debug,
      M : MyTrait<T>
{
    assert_eq!(m.get(), v);
}

pub fn main() {
    let value = MyType { dummy: 256 + 22 };
    test_eq::<usize, _>(value, value.dummy);
    test_eq::<u8, _>(value, value.dummy as u8);
}
