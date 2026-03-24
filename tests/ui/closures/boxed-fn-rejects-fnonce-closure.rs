//! Regression test for <https://github.com/rust-lang/rust/issues/26046>.
//!
//! Check that coercing a closure to `Box<dyn Fn() -> Vec<u8>>` rejects a
//! closure that only implements `FnOnce` because it moves a captured value out
//! of its environment.

fn get_closure() -> Box<dyn Fn() -> Vec<u8>> {
    let vec = vec![1u8, 2u8];

    let closure = move || { //~ ERROR expected a closure
        vec
    };

    Box::new(closure)
}

fn main() {}
