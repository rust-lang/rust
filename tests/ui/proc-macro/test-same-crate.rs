//@ compile-flags: --test
#![crate_type = "proc-macro"]

extern crate proc_macro;
use proc_macro::TokenStream;

#[proc_macro]
pub fn mac(input: TokenStream) -> TokenStream { loop {} }

#[cfg(test)]
mod test {
    #[test]
    fn t() { crate::mac!(A) }
    //~^ ERROR can't use a procedural macro from the same crate that defines it
    //~| HELP you can define integration tests in a directory named `tests`
}
