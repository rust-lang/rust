//@ revisions: cpass cpass2 cpass3

// This test checks that creating a new `DefId` from within a query `A`
// recreates that `DefId` before reexecuting queries that depend on query `A`.
// Otherwise we'd end up referring to a `DefId` that doesn't exist.
// At present this is handled by always marking all queries as red if they create
// a new `DefId` and thus subsequently rerunning the query.

trait Foo {
    fn foo() -> impl Sized;
}

#[cfg(any(cpass, cpass3))]
impl Foo for String {
    fn foo() -> i32 {
        22
    }
}

#[cfg(cpass2)]
impl Foo for String {
    fn foo() -> u32 {
        22
    }
}

fn main() {}
