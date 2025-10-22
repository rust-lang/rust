// Regression test for #130395, #146261 and #146586

enum A {
    B,
    C(),
    D(i32),
    E {},
    F { x: i32 },
}

fn foo(_: A) -> impl Fn() {
    || {}
}

fn main() {
    foo(A::b); //~ ERROR no variant or associated item named `b` found for enum `A` in the current scope [E0599]
    foo(A::c); //~ ERROR no variant or associated item named `c` found for enum `A` in the current scope [E0599]
    foo(A::d); //~ ERROR no variant or associated item named `d` found for enum `A` in the current scope [E0599]
    foo(A::e); //~ ERROR no variant or associated item named `e` found for enum `A` in the current scope [E0599]
    foo(A::f); //~ ERROR no variant or associated item named `f` found for enum `A` in the current scope [E0599]
    (A::b)(); //~ ERROR no variant or associated item named `b` found for enum `A` in the current scope [E0599]
    (A::c)(); //~ ERROR no variant or associated item named `c` found for enum `A` in the current scope [E0599]
    (A::d)(); //~ ERROR no variant or associated item named `d` found for enum `A` in the current scope [E0599]
    (A::e)(); //~ ERROR no variant or associated item named `e` found for enum `A` in the current scope [E0599]
    (A::f)(); //~ ERROR no variant or associated item named `f` found for enum `A` in the current scope [E0599]
    foo(A::b)(); //~ ERROR no variant or associated item named `b` found for enum `A` in the current scope [E0599]
    foo(A::c)(); //~ ERROR no variant or associated item named `c` found for enum `A` in the current scope [E0599]
    foo(A::d)(); //~ ERROR no variant or associated item named `d` found for enum `A` in the current scope [E0599]
    foo(A::e)(); //~ ERROR no variant or associated item named `e` found for enum `A` in the current scope [E0599]
    foo(A::f)(); //~ ERROR no variant or associated item named `f` found for enum `A` in the current scope [E0599]
}
