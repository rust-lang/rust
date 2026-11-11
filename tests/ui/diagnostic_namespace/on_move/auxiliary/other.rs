#![feature(diagnostic_on_move)]

#[diagnostic::on_move(
    message = "Foo",
    label = "Bar",
)]
#[derive(Debug)]
pub struct Foo;
