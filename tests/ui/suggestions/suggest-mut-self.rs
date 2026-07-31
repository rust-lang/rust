//! Regression test for <https://github.com/rust-lang/rust/issues/101145>.
//!
//! When an associated function forgets its `self` receiver but assigns *through* `self`, a plain
//! `&self` receiver would not compile, so `&mut self` is suggested instead. `self` that is only
//! read keeps the `&self` suggestion.

struct S {
    field: bool,
    count: u32,
    flags: [bool; 4],
}

impl S {
    // Assigned through, with a parameter to insert before.
    fn set(field: bool) {
        self.field = field;
        //~^ ERROR cannot find value `self` in this scope
    }

    // Assigned through, with no parameters: exercises the "insert after `(`" path.
    fn clear() {
        self.field = false;
        //~^ ERROR cannot find value `self` in this scope
    }

    // Parentheses around the receiver are transparent.
    fn set_paren() {
        (self).field = true;
        //~^ ERROR cannot find value `self` in this scope
    }

    // Writing through a deref of `self`.
    fn replace(other: S) {
        *self = other;
        //~^ ERROR cannot find value `self` in this scope
    }

    // Writing through an index, with the projection nested.
    fn set_flag() {
        self.flags[0] = true;
        //~^ ERROR cannot find value `self` in this scope
    }

    // Compound assignment writes through `self` too.
    fn bump() {
        self.count += 1;
        //~^ ERROR cannot find value `self` in this scope
    }

    // The left-hand `self` is written, the right-hand one is only read.
    fn dup() {
        self.field = self.flags[0];
        //~^ ERROR cannot find value `self` in this scope
        //~| ERROR cannot find value `self` in this scope
    }

    // Only read, so `&self` suffices.
    fn get() -> bool {
        self.field
        //~^ ERROR cannot find value `self` in this scope
    }

    // `self` is an operand of the index, not the base of the assigned place.
    fn store(arr: &mut [bool; 4]) {
        arr[self] = true;
        //~^ ERROR cannot find value `self` in this scope
    }
}

fn main() {}
