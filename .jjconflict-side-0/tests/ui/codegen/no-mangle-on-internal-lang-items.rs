// Issue a error when the user uses #[no_mangle] on internal language items
//@ edition:2024

#![feature(rustc_attrs)]

#[rustc_std_internal_symbol]
#[unsafe(no_mangle)] //~ERROR `#[no_mangle]` cannot be used on internal language items
fn internal_lang_function () {

}

fn main() {

}
