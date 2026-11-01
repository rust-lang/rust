#![feature(default_field_values)]

// If an API wants users to always use `..` even if they specify all the fields, they should use a
// sentinel field. As of now, that field can't be made private so it is only constructable with this
// syntax, but this might change in the future.

struct A {}
struct B();
struct C;
struct D {
    x: i32,
}
struct E(i32);

fn main() {
    let _ = A { .. }; //~ ERROR has no fields
    let _ = B { .. }; //~ ERROR has no fields
    let _ = C { .. }; //~ ERROR has no fields
    let _ = D { x: 0, .. };
    let _ = E { 0: 0, .. };
}
