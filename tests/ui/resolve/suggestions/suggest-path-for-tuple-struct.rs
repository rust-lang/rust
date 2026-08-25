mod module {
    pub struct SomeTupleStruct(u8);
    pub struct SomeRegularStruct {
        foo: u8
    }

    impl SomeTupleStruct {
        pub fn new() -> Self {
            Self(0)
        }
    }
    impl SomeRegularStruct {
        pub fn new() -> Self {
            Self { foo: 0 }
        }
    }
}

use module::{SomeTupleStruct, SomeRegularStruct};

fn main() {
    let _ = SomeTupleStruct.new();
    //~^ ERROR cannot find value `SomeTupleStruct` in this scope
    let _ = SomeRegularStruct.new();
    //~^ ERROR cannot find value `SomeRegularStruct` in this scope
}
