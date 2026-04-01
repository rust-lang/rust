// Test implicit coercions involving DSTs and raw pointers.

struct S;
trait T {}
impl T for S {}

struct Foo<T: ?Sized> {
    f: T
}

pub fn main() {
    // Test that we cannot convert from *-ptr to &S and &T
    let x: *const S = &S;
    let y: &S = x; //~ ERROR mismatched types
    let y: &dyn T = x; //~ ERROR mismatched types

    // Test that we cannot convert from *-ptr to &S and &T (mut version)
    let x: *mut S = &mut S;
    let y: &S = x; //~ ERROR mismatched types
    let y: &dyn T = x; //~ ERROR mismatched types

    // Test that we cannot convert an immutable ptr to a mutable one using *-ptrs
    let x: &mut dyn T = &S; //~ ERROR mismatched types
    let x: *mut dyn T = &S; //~ ERROR mismatched types
    let x: *mut S = &S; //~ ERROR mismatched types
}
