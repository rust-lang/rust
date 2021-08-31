#![feature(type_alias_impl_trait)]
#![allow(dead_code)]

mod m {
    use std::rc::Rc;

    type Foo = impl std::fmt::Debug;

    pub fn foo() -> Foo {
        Rc::new(22_u32)
    }
}

fn is_send<T: Send>(_: T) {}

fn main() {
    is_send(m::foo());
    //~^ ERROR: `Rc<u32>` cannot be sent between threads safely [E0277]
}
