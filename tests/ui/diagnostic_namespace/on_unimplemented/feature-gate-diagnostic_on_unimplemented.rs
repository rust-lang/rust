#[diagnostic::on_unimplemented(message = "Foo")]
//~^ERROR `#[diagnostic]` attribute name space is experimental [E0658]
pub trait Bar {
}

fn main() {
}
