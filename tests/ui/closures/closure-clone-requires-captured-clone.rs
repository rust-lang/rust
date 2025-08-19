//! Test that closures only implement `Clone` if all captured values implement `Clone`.
//!
//! When a closure captures variables from its environment, it can only be cloned
//! if all those captured variables are cloneable. This test makes sure the compiler
//! properly rejects attempts to clone closures that capture non-Clone types.

//@ compile-flags: --diagnostic-width=300

struct NonClone(i32);

fn main() {
    let captured_value = NonClone(5);
    let closure = move || {
        let _ = captured_value.0;
    };

    closure.clone();
    //~^ ERROR the trait bound `NonClone: Clone` is not satisfied
}
