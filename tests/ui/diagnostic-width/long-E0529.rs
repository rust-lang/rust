//@ compile-flags: --diagnostic-width=60 -Zwrite-long-types-to-disk=yes
//@ dont-require-annotations: NOTE

type A = (i32, i32, i32, i32);
type B = (A, A, A, A);
type C = (B, B, B, B);
type D = (C, C, C, C);

fn foo(x: D) {
    let [] = x; //~ ERROR expected an array or slice, found `(...
    //~^ NOTE pattern cannot match with input type `(...
}

fn main() {}
