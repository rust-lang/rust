//@ check-pass
//@ edition:2021
#![allow(todo_macro_calls)]

#![deny(opaque_hidden_inferred_bound)]

trait Repository /* : ?Sized */ {
    async fn new() -> Self;
}

struct MyRepository {}

impl Repository for MyRepository {
    async fn new() -> Self {
        todo!()
    }
}

fn main() {}
