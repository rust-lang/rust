//@ build-fail
//@ compile-flags:-Zpolymorphize=on
#![crate_type = "lib"]
#![feature(generic_const_exprs, rustc_attrs)]
//~^ WARN the feature `generic_const_exprs` is incomplete

#[rustc_polymorphize_error]
fn test<T>() {
    //~^ ERROR item has unused generic parameters
    let x = [0; 3 + 4];
}

pub fn caller() {
    test::<String>();
    test::<Vec<String>>();
}
