extern crate proc_macro;
use proc_macro::TokenStream;

// Adds `#[autodiff_reverse(<name>, Active, Active)]` to the item. The name is re-parsed, so it
// gets the call-site context, not the context of the user's tokens.
#[proc_macro_attribute]
pub fn add_autodiff(attr: TokenStream, item: TokenStream) -> TokenStream {
    let mut out: TokenStream =
        format!("#[::std::autodiff::autodiff_reverse({attr}, Active, Active)]").parse().unwrap();
    out.extend(item);
    out
}
