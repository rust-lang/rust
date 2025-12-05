//! Regression test for #118179: `adt_const_params` feature shouldn't leak
//! `{type error}` in error messages.

struct G<T, const N: Vec<T>>(T);
//~^ ERROR the type of const parameters must not depend on other generic parameters

fn main() {}
