trait Marker {}
impl Marker for u32 {}

trait MyTrait {
    fn foo(&self) -> impl Marker;
}

struct Outer;

impl MyTrait for Outer {
    fn foo(&self) -> impl Marker {
        42
    }
}

impl dyn MyTrait {
    //~^ ERROR the trait `MyTrait` cannot be made into an object
    fn other(&self) -> impl Marker {
        //~^ ERROR the trait `MyTrait` cannot be made into an object
        MyTrait::foo(&self)
        //~^ ERROR the trait bound `&dyn MyTrait: MyTrait` is not satisfied
        //~| ERROR the trait bound `&dyn MyTrait: MyTrait` is not satisfied
        //~| ERROR the trait `MyTrait` cannot be made into an object
    }
}

fn main() {}
