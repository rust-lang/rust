#[diagnostic::non_existing_attribute]
//~^ERROR `#[diagnostic]` attribute name space is experimental [E0658]
//~|WARNING unknown diagnostic attribute [unknown_or_malformed_diagnostic_attributes]
pub trait Bar {
}

#[diagnostic::non_existing_attribute(with_option = "foo")]
//~^ERROR `#[diagnostic]` attribute name space is experimental [E0658]
//~|WARNING unknown diagnostic attribute [unknown_or_malformed_diagnostic_attributes]
struct Foo;

fn main() {
}
