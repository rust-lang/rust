#![feature(custom_inner_attributes)]
#![feature(diagnostic_on_unknown)]
#![crate_type = "lib"]

#![diagnostic::on_unknown(message = "you silly, the crate `{This}` is empty")]
