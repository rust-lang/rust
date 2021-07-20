#![crate_type = "lib"]

const A: std::fmt::Arguments = format_args!("literal");

const B: std::fmt::Arguments = format_args!("{}", 123);
//~^ ERROR calls in constants are limited to
//~| ERROR calls in constants are limited to
//~| ERROR temporary value
