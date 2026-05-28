#![allow(dead_code, unused_variables)]
//@ run-rustfix
pub use my_mod::Foo;
//~^ NOTE the type is accessed through this re-export, but the type's constructor is not visible in this import's scope due to private fields
//~| NOTE the type is accessed through this re-export, but the type's constructor is not visible in this import's scope due to private fields

mod my_mod {
    pub struct Foo(u32);

    mod my_sub_mod {
        fn my_func() {
            let crate::Foo(x) = crate::Foo(42);
            //~^ ERROR cannot initialize a tuple struct which contains private fields
            //~| HELP the type can be constructed directly, because its fields are available from the current scope
            //~| ERROR cannot match against a tuple struct which contains private fields
            //~| HELP the type can be constructed directly, because its fields are available from the current scope
        }
    }
}
fn main() {}
