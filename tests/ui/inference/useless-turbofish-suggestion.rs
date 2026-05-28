// Regression test for #153732.
//
// When a method call already has turbofish type arguments, don't suggest
// rewriting them — the suggestion just rewrites user syntax into
// fully-qualified form without resolving anything.
//
// When a turbofish is present, the diagnostic points at the specific
// uninferred generic argument rather than the method name.

struct S;

impl S {
    fn f<A, B>(self, _a: A) -> B {
        todo!()
    }
}

fn turbofish_second_arg() {
    S.f::<u32, _>(42);
    //~^ ERROR type annotations needed
}

fn turbofish_first_arg() {
    S.f::<_, _>(42);
    //~^ ERROR type annotations needed
}

fn without_turbofish() {
    S.f(42);
    //~^ ERROR type annotations needed
}

fn main() {}
