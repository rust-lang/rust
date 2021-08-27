#![feature(const_param_types)]
//~^ WARN the feature `const_param_types` is incomplete

fn function_with_str<'a, const STRING: &'a str>() {} //~ ERROR E0771

fn main() {
    function_with_str::<"Hello, world!">()
}
