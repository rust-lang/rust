#![feature(lint_reasons)]

#![expect(reason = "This should trigger because `unused_mut` was allow", unused_mut)]
//~^ ERROR malformed lint attribute
//~| ERROR malformed lint attribute
//~| NOTE reason in lint attribute must come last
//~| NOTE reason in lint attribute must come last

fn main() {}
