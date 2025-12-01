#![feature(extern_item_impls)]
// Check whether the EII attributes do target checking properly.

#[eii]
fn foo() {}

#[foo]
//~^ ERROR `#[foo]` is only valid on functions
#[eii]
//~^ ERROR `#[eii]` is only valid on functions
const A: usize = 3;

#[foo]
//~^ ERROR `#[foo]` is only valid on functions
#[eii]
//~^ ERROR `#[eii]` is only valid on functions
macro_rules! foo_impl {
    () => {};
}

#[foo]
//~^ ERROR `#[foo]` is only valid on functions
#[eii]
//~^ ERROR `#[eii]` is only valid on functions
struct Foo;

#[foo]
//~^ ERROR `#[foo]` is only valid on functions
#[eii]
//~^ ERROR `#[eii]` is only valid on functions
impl Foo {
    #[foo]
    //~^ ERROR `#[foo]` is only valid on functions
    #[eii]
    //~^ ERROR `#[eii]` is only valid on functions
    fn foo_impl() {}
}

#[foo]
//~^ ERROR `#[foo]` is only valid on functions
#[eii]
//~^ ERROR `#[eii]` is only valid on functions
trait Bar {
    #[foo]
    //~^ ERROR `#[foo]` is only valid on functions
    #[eii]
    //~^ ERROR `#[eii]` is only valid on functions
    fn foo_impl();
}

#[foo]
//~^ ERROR `#[foo]` is only valid on functions
#[eii]
//~^ ERROR `#[eii]` is only valid on functions
impl Bar for Foo {
    #[foo]
    //~^ ERROR `#[foo]` is only valid on functions
    #[eii]
    //~^ ERROR `#[eii]` is only valid on functions
    fn foo_impl() {}
}

fn main() {}
