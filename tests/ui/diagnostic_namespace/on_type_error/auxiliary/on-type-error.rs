#![feature(diagnostic_on_type_error)]
#[diagnostic::on_type_error(
    note = "custom on_type_error note: expected struct `{expected}`\n found struct `{found}`"
)]
pub struct S<T>(pub T);
