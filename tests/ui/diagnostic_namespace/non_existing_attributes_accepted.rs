#![feature(diagnostic_namespace)]
// check-pass
#[diagnostic::non_existing_attribute]
//~^WARN unknown diagnostic attribute
pub trait Bar {
}

#[diagnostic::non_existing_attribute(with_option = "foo")]
//~^WARN unknown diagnostic attribute
struct Foo;

fn main() {
}
