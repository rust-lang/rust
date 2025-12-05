//@ compile-flags: --diagnostic-width=60 -Zwrite-long-types-to-disk=yes

type A = (i32, i32, i32, i32);
type B = (A, A, A, A);
type C = (B, B, B, B);
type D = (C, C, C, C);

fn foo(x: D) {
    x.field; //~ ERROR no field `field` on type `(...
}

fn main() {}
