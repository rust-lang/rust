pub struct Foo;

#[diagnostic::on_const(message = "tadaa", note = "boing")]
impl PartialEq for Foo {
    fn eq(&self, _other: &Foo) -> bool {
        true
    }
}
