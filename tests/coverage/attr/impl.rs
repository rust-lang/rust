#![feature(coverage_attribute)]
//@ edition: 2021
//@ reference: attributes.coverage.nesting

// Checks that `#[coverage(..)]` can be applied to impl and impl-trait blocks,
// and is inherited by any enclosed functions.

struct MyStruct;

#[coverage(off)]
impl MyStruct {
    fn off_inherit() {}

    #[coverage(on)]
    fn off_on() {}

    #[coverage(off)]
    fn off_off() {}
}

#[coverage(on)]
impl MyStruct {
    fn on_inherit() {}

    #[coverage(on)]
    fn on_on() {}

    #[coverage(off)]
    fn on_off() {}
}

trait MyTrait {
    fn method();
}

#[coverage(off)]
impl MyTrait for MyStruct {
    fn method() {}
}

#[coverage(off)]
fn main() {}
