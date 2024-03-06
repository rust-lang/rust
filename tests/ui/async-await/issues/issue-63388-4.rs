//@ check-pass
//@ edition:2018

struct A;

impl A {
    async fn foo(&self, f: &u32) -> &A { self }
}

fn main() { }
