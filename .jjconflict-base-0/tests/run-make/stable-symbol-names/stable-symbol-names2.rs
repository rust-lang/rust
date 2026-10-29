#![crate_type = "rlib"]

extern crate stable_symbol_names1;

pub fn user() {
    stable_symbol_names1::generic_function(1u32);
    stable_symbol_names1::generic_function("def");
    let x = 2u64;
    stable_symbol_names1::generic_function(&x);
    stable_symbol_names1::mono_function();
    stable_symbol_names1::mono_function_lifetime(&0);
}

pub fn trait_impl_test_function() {
    use stable_symbol_names1::*;
    Bar::generic_method::<Bar>();
}
