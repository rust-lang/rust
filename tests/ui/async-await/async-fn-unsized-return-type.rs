//@ edition: 2021
#![deny(opaque_hidden_inferred_bound)]
// Test that async functions cannot return unsized types via `impl Trait + ?Sized`
// Issue #149438

use std::fmt::Debug;

async fn unsized_async() -> impl Debug + ?Sized {
    //~^ ERROR opaque type `impl Future<Output = impl Debug + ?Sized>` does not satisfy its associated type bounds
    123
}

fn main() {}
