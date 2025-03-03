#![warn(clippy::pedantic)]
// Should suggest lowercase
#![allow(clippy::All)]
//~^ ERROR: unknown lint
#![warn(clippy::CMP_OWNED)]
//~^ ERROR: unknown lint

// Should suggest similar clippy lint name
#[warn(clippy::if_not_els)]
//~^ ERROR: unknown lint
#[warn(clippy::UNNecsaRy_cAst)]
//~^ ERROR: unknown lint
#[warn(clippy::useles_transute)]
//~^ ERROR: unknown lint
// Should suggest rustc lint name(`dead_code`)
#[warn(clippy::dead_cod)]
//~^ ERROR: unknown lint
// Shouldn't suggest removed/deprecated clippy lint name(`unused_collect`)
#[warn(clippy::unused_colle)]
//~^ ERROR: unknown lint
// Shouldn't suggest renamed clippy lint name(`const_static_lifetime`)
#[warn(clippy::const_static_lifetim)]
//~^ ERROR: unknown lint
// issue #118183, should report `missing_docs` from rustc lint
#[warn(clippy::missing_docs)]
//~^ ERROR: unknown lint
fn main() {}
