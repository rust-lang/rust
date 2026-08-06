// This tests the extra note reported when a type error deals with
// seemingly identical types.
// The main use case of this error is when there are two crates
// (generally different versions of the same crate) with the same name
// causing a type mismatch.

// The test is nearly the same as the one in
// ui/type/type-mismatch-same-crate-name.rs
// but deals with the case where one of the crates
// is only introduced as an indirect dependency.
// and the type is accessed via a re-export.
// This is similar to how the error can be introduced
// when using cargo's automatic dependency resolution.

extern crate crateA;

fn main() {
    let foo2 = crateA::Foo;
    let bar2 = crateA::bar();
    {
        extern crate crateB;
        crateB::try_foo(foo2);
        crateB::try_bar(bar2);
    }
}
