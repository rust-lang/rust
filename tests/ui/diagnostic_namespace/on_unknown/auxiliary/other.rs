#![crate_type = "lib"]

#[diagnostic::on_unknown(message = "you silly, this module is empty")]
pub mod empty {}
