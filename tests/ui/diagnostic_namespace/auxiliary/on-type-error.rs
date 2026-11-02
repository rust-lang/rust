#![feature(diagnostic_on_type_error)]

#[diagnostic::on_type_error(
    note = "[TEST] cross crate expected `{expected}`, found `{found}`"
)]
pub struct Foo<T>(pub T);
