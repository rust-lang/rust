#![warn(useless_send_constraint)]

use std::any::Any;

fn main() {}

fn fine(_a: &dyn Any) {}

fn should_replace_with_any(_a: &(dyn Send)) {}

fn should_remove_send(_a: &(dyn Any + Send)) {}

fn should_remove_send_duplicate(_a: &(dyn Any + Send)) {}
