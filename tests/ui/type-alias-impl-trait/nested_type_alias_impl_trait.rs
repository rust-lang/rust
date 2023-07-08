#![feature(type_alias_impl_trait)]

mod my_mod {
    use std::fmt::Debug;

    pub type Foo = impl Debug;
    pub type Foot = impl Debug;

    pub fn get_foo() -> Foo {
        5i32
    }

    pub fn get_foot(_: Foo) -> Foot {
        get_foo() //~ ERROR opaque type's hidden type cannot be another opaque type
    }
}

fn main() {
    let _: my_mod::Foot = my_mod::get_foot(my_mod::get_foo());
}
