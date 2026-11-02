extern crate proc_macro;
use proc_macro::TokenStream;

#[proc_macro_attribute]
pub fn phantom(_: TokenStream, _: TokenStream) -> TokenStream {
    "
mod __value_Spooky {
    pub use super::Spooky::Spooky;
}
pub enum Spooky<T> {
    __Phantom(std::marker::PhantomData<T>),
    Spooky,
}
pub use self::__value_Spooky::*;
    "
    .parse()
    .unwrap()
}
