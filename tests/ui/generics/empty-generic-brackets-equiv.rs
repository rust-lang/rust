//! Test that empty type parameter list <> is equivalent to no type parameters
//!
//! Checks` that empty angle brackets <> are syntactically valid and equivalent
//! to omitting type parameters entirely across various language constructs.

//@ run-pass

struct S<>;
trait T<> {} //~ WARN trait `T` is never used
enum E<> {
    V
}
impl<> T<> for S<> {}
impl T for E {}
fn foo<>() {}
fn bar() {}
fn main() {
    let _ = S;
    let _ = S::<>;
    let _ = E::V;
    let _ = E::<>::V;
    foo();
    foo::<>();
    // Test that we can supply <> to non-generic things
    bar::<>();
    let _: i32<>;
}
