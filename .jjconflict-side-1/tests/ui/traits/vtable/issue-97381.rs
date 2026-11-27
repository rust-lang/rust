use std::ops::Deref;
trait MyTrait: Deref<Target = u32> {}
struct MyStruct(u32);
impl MyTrait for MyStruct {}
impl Deref for MyStruct {
    type Target = u32;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
fn get_concrete_value(i: u32) -> MyStruct {
    MyStruct(i)
}
fn get_boxed_value(i: u32) -> Box<dyn MyTrait> {
    Box::new(get_concrete_value(i))
}
fn main() {
    let v = [1, 2, 3]
        .iter()
        .map(|i| get_boxed_value(*i))
        .collect::<Vec<_>>();

    let el = &v[0];

    for _ in v {
        //~^ ERROR cannot move out of `v` because it is borrowed
        println!("{}", ***el > 0);
    }
}
