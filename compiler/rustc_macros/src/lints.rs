use quote::{format_ident, quote};
use syn::{ImplItem, ItemImpl, parse_macro_input, parse_quote};

pub(crate) fn runtime_lint_pass(
    attr: proc_macro::TokenStream,
    input: proc_macro::TokenStream,
) -> proc_macro::TokenStream {
    if !attr.is_empty() {
        panic!("`#[runtime_lint_pass]` cannot take an argument");
    }

    let mut item = parse_macro_input!(input as ItemImpl);

    let needed_methods: Vec<ImplItem> = item
        .items
        .iter()
        .filter_map(|item| {
            match item {
                // We are looking for `check_*` functions.
                ImplItem::Fn(f) => {
                    let name = &f.sig.ident;
                    if !name.to_string().starts_with("check_") {
                        panic!("`#[runtime_lint_pass]` found fn name not matching `check_*`");
                    }
                    let needed_name = format_ident!("{}_needed", name);
                    Some(parse_quote! {
                        fn #needed_name(&self) -> bool { true }
                    })
                }

                // Some clippy lints use a macro to provide some of the `check_*` fns. Allow and
                // ignore them.
                ImplItem::Macro(_) => None,

                // Other item kinds (e.g. types, consts) should not appear.
                _ => panic!("`#[runtime_lint_pass]` found a non-fn, non-macro item"),
            }
        })
        .collect();

    item.items.extend(needed_methods);

    quote! { #item }.into()
}
