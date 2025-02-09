pub mod foo {
    pub struct Bar;
}

#[diagnostic::on_unknown_item(
    message = "first message",
    label = "first label",
    note = "custom note",
    note = "custom note 2"
)]
use foo::Foo;
//~^ERROR first message

use foo::Bar;

fn main() {}
