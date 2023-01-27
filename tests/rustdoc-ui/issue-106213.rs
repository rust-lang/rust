// compile-flags: --document-private-items
// edition:2021

fn use_avx() -> dyn  {
    !(ident_error)
    //~^ ERROR cannot find value `ident_error` in this scope
}
