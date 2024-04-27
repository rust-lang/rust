#![feature(adt_const_params)]
//~^ WARN the feature `adt_const_params` is incomplete

fn function_with_str<'a, const STRING: &'a str>() {} //~ ERROR E0770

fn main() {
    function_with_str::<"Hello, world!">()
}
