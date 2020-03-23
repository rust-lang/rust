#![allow(rustc::default_hash_types)]
#![recursion_limit = "128"]

use synstructure::decl_derive;

use proc_macro::TokenStream;

mod hash_stable;
mod lift;
mod query;
mod symbols;
mod type_foldable;

#[proc_macro]
pub fn rustc_queries(input: TokenStream) -> TokenStream {
    query::rustc_queries(input)
}

#[proc_macro]
pub fn symbols(input: TokenStream) -> TokenStream {
    symbols::symbols(input)
}

decl_derive!([HashStable, attributes(stable_hasher)] => hash_stable::hash_stable_derive);
decl_derive!(
    [HashStable_Generic, attributes(stable_hasher)] =>
    hash_stable::hash_stable_generic_derive
);

decl_derive!([TypeFoldable, attributes(type_foldable)] => type_foldable::type_foldable_derive);
decl_derive!([Lift, attributes(lift)] => lift::lift_derive);
