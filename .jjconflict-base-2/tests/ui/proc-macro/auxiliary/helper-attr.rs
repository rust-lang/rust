extern crate proc_macro;

// Doesn't do anything, but has a helper attribute.
#[proc_macro_derive(WithHelperAttr, attributes(x))]
pub fn derive(_input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    proc_macro::TokenStream::new()
}
