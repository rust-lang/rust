//@ edition: 2021
//@ check-pass

// Test that `'r#a` is `'a`.

fn test<'r#a>(x: &'a ()) {}

fn main() {}
