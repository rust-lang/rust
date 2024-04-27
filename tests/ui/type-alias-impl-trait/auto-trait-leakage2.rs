#![feature(type_alias_impl_trait)]
#![allow(dead_code)]

mod m {
    use std::rc::Rc;

    type Foo = impl std::fmt::Debug; //~ NOTE appears within the type
    //~^ within this `Foo`
    //~| expansion of desugaring

    pub fn foo() -> Foo {
        Rc::new(22_u32)
    }
}

fn is_send<T: Send>(_: T) {}
//~^ required by this bound
//~| required by a bound

fn main() {
    is_send(m::foo());
    //~^ ERROR: `Rc<u32>` cannot be sent between threads safely [E0277]
    //~| NOTE cannot be sent
    //~| NOTE required by a bound
}
