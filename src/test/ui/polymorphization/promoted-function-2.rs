// build-fail
// compile-flags:-Zpolymorphize=on
#![crate_type = "lib"]
#![feature(lazy_normalization_consts, rustc_attrs)]
//~^ WARN the feature `lazy_normalization_consts` is incomplete

#[rustc_polymorphize_error]
fn test<T>() {
    //~^ ERROR item has unused generic parameters
    let x = [0; 3 + 4];
}

pub fn caller() {
    test::<String>();
    test::<Vec<String>>();
}
