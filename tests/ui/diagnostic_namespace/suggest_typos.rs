//@ reference: attributes.diagnostic.namespace.unknown-invalid-syntax
#![deny(unknown_diagnostic_attributes)]

#[diagnostic::onunimplemented]
//~^ERROR unknown diagnostic attribute
//~^^HELP an attribute with a similar name exists
trait X{}

#[diagnostic::un_onimplemented]
//~^ERROR unknown diagnostic attribute
//~^^HELP an attribute with a similar name exists
trait Y{}

#[diagnostic::on_implemented]
//~^ERROR unknown diagnostic attribute
//~^^HELP an attribute with a similar name exists
trait Z{}

#[diagnostic::dont_recommend]
//~^ERROR unknown diagnostic attribute
//~^^HELP an attribute with a similar name exists
impl X for u8 {}

fn main(){}
