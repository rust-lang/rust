pub struct MyStruct {}

mod submodule {
    use super::MyStruct;

    pub struct MyRef<'a>(&'a MyStruct);
}

use self::submodule::MyRef;

impl MyStruct {
    pub fn method(&self) -> MyRef {
        MyRef(self)
        //~^ ERROR cannot initialize a tuple struct which contains private fields
    }
}

fn main() {}
