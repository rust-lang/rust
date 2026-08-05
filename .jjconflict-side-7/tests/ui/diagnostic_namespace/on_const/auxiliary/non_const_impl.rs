#![feature(diagnostic_on_const)]

pub struct X;

#[diagnostic::on_const(
    message = "their message",
    label = "their label",
    note = "their note",
    note = "their other note"
)]
impl PartialEq for X {
    fn eq(&self, _other: &X) -> bool {
        true
    }
}
