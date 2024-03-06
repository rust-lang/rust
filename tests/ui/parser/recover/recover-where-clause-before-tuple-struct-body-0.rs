// Regression test for issues #100790 and #106439.
//@ run-rustfix

pub struct Example
where
    (): Sized,
(#[allow(dead_code)] usize);
//~^^^ ERROR where clauses are not allowed before tuple struct bodies

struct _Demo
where
    (): Sized,
    String: Clone,
(pub usize, usize);
//~^^^^ ERROR where clauses are not allowed before tuple struct bodies

fn main() {}
