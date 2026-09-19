//@ compile-flags: --diagnostic-width=60 -Zwrite-long-types-to-disk=yes
//@ dont-require-annotations: NOTE

type A = (i32, i32, i32, i32);
type B = (A, A, A, A);
type C = (B, B, B, B);
type D = (C, C, C, C);

fn foo(x: D) { //~ NOTE `x` has type `((_
    x(); //~ ERROR expected function, found `((_
}

fn main() {}
