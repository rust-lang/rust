//@ compile-flags: --document-private-items
//@ edition:2021
// https://github.com/rust-lang/rust/issues/106213

fn use_avx() -> dyn  {
    //~^ ERROR at least one trait is required for an object type
    !( ident_error )
}
