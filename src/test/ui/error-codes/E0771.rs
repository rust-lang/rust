#![feature(const_generics)]
//~^ WARN the feature `const_generics` is incomplete

fn function_with_str<'a, const STRING: &'a str>() {} //~ ERROR E0771

fn main() {
    function_with_str::<"Hello, world!">()
}
