#![feature(type_alias_impl_trait)]

mod my_mod {
    use std::fmt::Debug;

    pub type Foo = impl Debug;
    pub type Foot = impl Debug;

    #[defines(Foo)]
    pub fn get_foo() -> Foo {
        5i32
    }

    // remove the `defines(Foo)` to make it unambiguous and pass
    #[defines(Foo)]
    #[defines(Foot)]
    pub fn get_foot() -> Foot {
        get_foo() //~ ERROR opaque type's hidden type cannot be another opaque type
    }
}

fn main() {
    let _: my_mod::Foot = my_mod::get_foot();
}
