//@rustc-env: CLIPPY_PETS_PRINT=1
#![feature(lint_reasons)]

#[warn(clippy::borrow_pats)]
fn main() {}
