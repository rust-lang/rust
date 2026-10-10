//@ check-pass
//@ compile-flags: -Znext-solver

// A regression test for https://github.com/rust-lang/trait-system-refactor-initiative/issues/266.
// Ensure that we do not accidentaly trying unfulfilled unsized coercions due to hitting recursion
// limits while trying to find the right fulfillment error source.

fn argument_coercion<U>(_: &U) {}

pub fn test() {
    argument_coercion(&{
        Nested(0.0, 0.0)
            .add(0.0)
            .add(0.0)
            .add(0.0)
            .add(0.0)
            .add(0.0)
            .add(0.0)
            .add(0.0)
            .add(0.0)
            .add(0.0)
            .add(0.0)
            .add(0.0)
    });
}

struct Nested<T, R>(T, R);

impl<T, R> Nested<T, R> {
    fn add<U>(self, value: U) -> Nested<U, Nested<T, R>> {
        Nested(value, self)
    }
}

fn main() {}
