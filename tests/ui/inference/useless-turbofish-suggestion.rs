// Regression test for #153732.
//
// When a method call already has turbofish type arguments, don't suggest
// rewriting them — the suggestion just rewrites user syntax into
// fully-qualified form without resolving anything.
//
// The span still points at the method name rather than the unresolved `_`;
// fixing that is left as future work.

struct S;

impl S {
    fn f<A, B>(self, _a: A) -> B {
        todo!()
    }
}

fn with_turbofish() {
    S.f::<u32, _>(42);
    //~^ ERROR type annotations needed
}

fn without_turbofish() {
    S.f(42);
    //~^ ERROR type annotations needed
}

fn main() {}
