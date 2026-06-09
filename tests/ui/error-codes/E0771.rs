#![feature(adt_const_params, unsized_const_params)]

fn function_with_str<'a, const STRING: &'a str>() {} //~ ERROR E0770

fn main() {
    function_with_str::<"Hello, world!">()
}
