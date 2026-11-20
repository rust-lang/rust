#![feature(diagnostic_on_type_error)]

#[diagnostic::on_type_error(
    note = "[TEST] cross crate expected `{Expected}`, found `{Found}`"
)]
pub struct Foo<T>(pub T);
