//@ run-pass

#![feature(arbitrary_self_types)]
#![allow(dead_code)]

// With arbitrary self types v2, we show an error if there are
// multiple contenders for a method call in an inner and outer type.
// The goal is to avoid any possibility of confusion by a new
// 'shadowing' method calling a 'shadowed' method.
// However, there are niche circumstances where this
// algorithm doesn't quite work, due to reborrows to get a different
// lifetime. The test below explicitly tests those circumstances to ensure
// the behavior is as expected, even if it's not 100% desirable. They're
// very niche circumstances.

#[derive(Debug, PartialEq)]
enum Callee {
    INNER,
    OUTER
}

struct MyNonNull<T>(T);

impl<T> std::ops::Receiver for MyNonNull<T> {
    type Target = T;
}

struct A;
impl A {
    fn foo(self: MyNonNull<A>) -> Callee {
        Callee::INNER
    }

    fn bar(self: &MyNonNull<A>) -> Callee {
        Callee::INNER
    }

    fn baz(self: &&MyNonNull<A>) -> Callee {
        // note this is by DOUBLE reference
        Callee::INNER
    }
}

impl<T> MyNonNull<T> {
    fn foo(&self) -> Callee{
        Callee::OUTER
    }

    fn bar(&self) -> Callee{
        Callee::OUTER
    }

    fn baz(&self) -> Callee{
        Callee::OUTER
    }
}

fn main() {
    // The normal deshadowing case. Does not compile.
    // assert_eq!(MyNonNull(A).foo(), Callee::INNER);

    // Similarly, does not compile.
    //assert_eq!(MyNonNull(A).bar(), Callee::INNER);

    // The double-reference case.
    // We call the newly-added outer type method.
    // Not ideal but very niche so we accept it.
    assert_eq!(MyNonNull(A).baz(), Callee::OUTER);
}
