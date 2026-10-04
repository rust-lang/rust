#![feature(custom_inner_attributes)]
#![crate_type = "lib"]

#![diagnostic::on_unknown(message = "you silly, the crate `{This}` is empty")]
