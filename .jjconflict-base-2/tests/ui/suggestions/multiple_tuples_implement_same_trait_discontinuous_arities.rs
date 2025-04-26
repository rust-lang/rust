fn testing_mytrait<T: MyTrait>(t: T) {}
fn testing_anothertrait<T: AnotherTrait>(t: T) {}

trait MyTrait {}

impl MyTrait for (i8,) {}
impl MyTrait for (i8,i8,i8) {}
impl MyTrait for (i8,i8,i8,i8) {}
impl MyTrait for (i8,i8,i8,i8,i8) {}
impl MyTrait for (i8,i8,i8,i8,i8,i8) {}

trait AnotherTrait {}
impl AnotherTrait for (i8,) {}
impl AnotherTrait for (i8,i8) {}

struct Foo;

fn main() {
    testing_mytrait((1, Foo));
    //~^ ERROR the trait bound `({integer}, Foo): MyTrait` is not satisfied [E0277]
    testing_anothertrait((1, Foo));
    //~^ ERROR the trait bound `({integer}, Foo): AnotherTrait` is not satisfied [E0277]
}
