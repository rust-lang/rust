#![feature(diagnostic_on_type_error)]

#[diagnostic::on_type_error]
struct S<T>(T);

fn main() {}
