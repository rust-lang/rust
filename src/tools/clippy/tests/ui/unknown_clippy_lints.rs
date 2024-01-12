#![warn(clippy::pedantic)]
// Should suggest lowercase
#![allow(clippy::All)]
#![warn(clippy::CMP_OWNED)]

// Should suggest similar clippy lint name
#[warn(clippy::if_not_els)]
#[warn(clippy::UNNecsaRy_cAst)]
#[warn(clippy::useles_transute)]
// Should suggest rustc lint name(`dead_code`)
#[warn(clippy::dead_cod)]
// Shouldn't suggest removed/deprecated clippy lint name(`unused_collect`)
#[warn(clippy::unused_colle)]
// Shouldn't suggest renamed clippy lint name(`const_static_lifetime`)
#[warn(clippy::const_static_lifetim)]
// issue #118183, should report `missing_docs` from rustc lint
#[warn(clippy::missing_docs)]
fn main() {}
