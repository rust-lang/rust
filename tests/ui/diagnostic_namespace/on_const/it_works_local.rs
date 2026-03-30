#![crate_type = "lib"]
#![feature(diagnostic_on_const)]

pub struct X;

#[diagnostic::on_const(
    message = "my message",
    label = "my label",
    note = "my note",
    note = "my other note"
)]
impl PartialEq for X {
    //~^NOTE: my label
    fn eq(&self, _other: &X) -> bool {
        true
    }
}

const _: () = {
    let x = X;
    x == x;
    //~^ ERROR: my message
    //~| NOTE: my note
    //~| NOTE: my other note
};
