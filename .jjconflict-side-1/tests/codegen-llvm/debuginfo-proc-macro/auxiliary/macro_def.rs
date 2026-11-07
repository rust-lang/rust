extern crate proc_macro;
use proc_macro::*;

#[proc_macro]
pub fn square_twice(_item: TokenStream) -> TokenStream {
    "(square(env::vars().count() as i32), square(env::vars().count() as i32))".parse().unwrap()
}
