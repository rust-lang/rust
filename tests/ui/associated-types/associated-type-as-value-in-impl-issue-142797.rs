// issue: <https://github.com/rust-lang/rust/issues/142797>

#![allow(dead_code, unused_variables)]

struct SomeStruct {
    some_field: u64,
}

trait SomeTrait {
    type SomeType;
    fn foo();
}

impl SomeTrait for bool {
    type SomeType = SomeStruct;

    fn foo() {
        let ss = Self::SomeType;
        //~^ ERROR no associated function or constant named `SomeType` found for type `bool` in the current scope
    }
}

fn main() {}
