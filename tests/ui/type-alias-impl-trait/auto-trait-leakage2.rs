#![feature(type_alias_impl_trait)]
#![allow(dead_code)]

use std::rc::Rc;

type Foo = impl std::fmt::Debug; //~ NOTE appears within the type
//~^ NOTE within this `Foo`
//~| NOTE expansion of desugaring

#[define_opaque(Foo)]
pub fn foo() -> Foo {
    Rc::new(22_u32)
}

fn is_send<T: Send>(_: T) {}
//~^ NOTE required by this bound
//~| NOTE required by a bound

fn main() {
    is_send(foo());
    //~^ ERROR: `Rc<u32>` cannot be sent between threads safely [E0277]
    //~| NOTE cannot be sent
    //~| NOTE required by a bound
}
