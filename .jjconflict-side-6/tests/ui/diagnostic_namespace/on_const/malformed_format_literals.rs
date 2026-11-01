#![crate_type = "lib"]
#![feature(diagnostic_on_const)]
#![feature(const_trait_impl)]

pub struct X;

const trait Y {
    fn blah(&self) {}
}

#[diagnostic::on_const(
    message = "my message {Foo}",
    //~^ WARN unknown parameter `Foo`
    label = "my label {Bar}",
    //~^ WARN unknown parameter `Bar`
    note = "my label {Baz}",
    //~^ WARN unknown parameter `Baz`
)]
impl Y for X {}

const _: () = {
    X {}.blah();
    //~^ ERROR my message {Foo} [E0277]

};
