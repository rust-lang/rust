


// This tests that exports can have visible dependencies on things
// that are not exported, allowing for a sort of poor-man's ADT
mod foo {

    // not exported
    tag t { t1; }
    fn f() -> t { ret t1; }
    fn g(t v) { assert (v == t1); }
}

fn main() { foo::g(foo::f()); }