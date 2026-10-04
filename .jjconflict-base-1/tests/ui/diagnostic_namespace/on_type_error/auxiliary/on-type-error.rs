#![feature(diagnostic_on_type_error)]
#[diagnostic::on_type_error(
    note = "custom on_type_error note: expected struct `{Expected}`\n found struct `{Found}`"
)]
pub struct S<T>(pub T);
