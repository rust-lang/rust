// run-pass
trait TheTrait : TheSuperTrait<<Self as TheTrait>::Item> {
    type Item;
}

trait TheSuperTrait<T> {
    fn get(&self) -> T;
}

impl TheTrait for i32 {
    type Item = u32;
}

impl TheSuperTrait<u32> for i32 {
    fn get(&self) -> u32 {
        *self as u32
    }
}

fn foo<T:TheTrait<Item=u32>>(t: &T) -> u32 {
    t.get()
}

fn main() {
    foo::<i32>(&22);
}
