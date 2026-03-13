#![feature(diagnostic_on_const)]

pub struct X;

#[diagnostic::on_const(message = "message", label = "label", note = "note")]
impl PartialEq for X {
    fn eq(&self, _other: &X) -> bool {
        true
    }
}
