//@ run-pass
// Check that closures implement `TrivialClone`.

#![feature(trivial_clone)]

use std::clone::TrivialClone;

fn require_trivial_clone<T: TrivialClone>(_t: T) {}

fn main() {
    let some_trivial_clone_value = 42i32;
    require_trivial_clone(move || some_trivial_clone_value);
}
