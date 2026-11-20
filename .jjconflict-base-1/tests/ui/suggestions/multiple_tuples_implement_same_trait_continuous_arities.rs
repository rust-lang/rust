use std::fmt::Debug;

fn testing_debug<T: Debug>(t: T) {}
fn testing_mytrait<T: MyTrait>(t: T) {}

trait MyTrait {}

impl MyTrait for (i8,) {}
impl MyTrait for (i8,i8) {}
impl MyTrait for (i8,i8,i8) {}
impl MyTrait for (i8,i8,i8,i8) {}
impl MyTrait for (i8,i8,i8,i8,i8) {}

struct Foo;

fn main() {
    testing_debug((1, Foo));
    //~^ ERROR `Foo` doesn't implement `Debug` [E0277]
    testing_mytrait((1, Foo));
    //~^ ERROR the trait bound `({integer}, Foo): MyTrait` is not satisfied [E0277]
}
