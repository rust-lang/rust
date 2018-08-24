#![feature(associated_consts, rustc_attrs)]
#![allow(warnings)]

trait MyTrait {
    const MY_CONST: &'static str;
}

macro_rules! my_macro {
    () => {
        struct MyStruct;

        impl MyTrait for MyStruct {
            const MY_CONST: &'static str = stringify!(abc);
        }
    }
}

my_macro!();

#[rustc_error]
fn main() {} //~ ERROR compilation successful
