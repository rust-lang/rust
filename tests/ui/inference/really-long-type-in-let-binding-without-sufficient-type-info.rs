//@ compile-flags: -Zwrite-long-types-to-disk=yes
type A = (i32, i32, i32, i32);
type B = (A, A, A, A);
type C = (B, B, B, B);
type D = (C, C, C, C);

fn foo(x: D) {
    let y = Err(x); //~ ERROR type annotations needed for `Result<_
}

fn main() {}
