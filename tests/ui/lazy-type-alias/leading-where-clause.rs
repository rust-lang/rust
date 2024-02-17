//@ run-rustfix

#![feature(lazy_type_alias)]
#![allow(incomplete_features)]

// Check that we *reject* leading where-clauses on lazy type aliases.

type Alias<T>
where
    String: From<T>,
= T;
//~^^^ ERROR where clauses are not allowed before the type for type aliases

fn main() {
    let _: Alias<&str>;
}
