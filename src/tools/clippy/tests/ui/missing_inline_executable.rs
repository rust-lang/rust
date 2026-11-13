#![warn(clippy::missing_inline_in_public_items)]

pub fn foo() {}
//~^ missing_inline_in_public_items

fn main() {}
