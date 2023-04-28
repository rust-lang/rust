// check-pass
#![warn(useless_send_constraint)]

use std::any::Any;

fn main() {}

fn foo(_a: &(dyn Any + Send + Send)) {

}
