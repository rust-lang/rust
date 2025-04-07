//@ compile-flags: --diagnostic-width=100 -Zwrite-long-types-to-disk=yes

type A = (i32, i32, i32, i32);
type B = (A, A, A, A);
type C = (B, B, B, B);
type D = (C, C, C, C);

fn foo(x: D) {
    let () = x; //~ ERROR mismatched types
    //~^ NOTE this expression has type `((...,
    //~| NOTE expected `((...,
    //~| NOTE expected tuple
    //~| NOTE the full name for the type has been written to
    //~| NOTE consider using `--verbose` to print the full type name to the console
}

fn main() {}
