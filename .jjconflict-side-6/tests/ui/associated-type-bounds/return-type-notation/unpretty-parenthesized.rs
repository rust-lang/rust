//@ edition: 2021
//@ compile-flags: -Zunpretty=expanded
//@ check-pass

// NOTE: This is not considered RTN syntax currently.
// This is simply parenthesized generics.

trait Trait {
    async fn method() {}
}

fn foo<T: Trait<method(i32): Send>>() {}

fn main() {}
