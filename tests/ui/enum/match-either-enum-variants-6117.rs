// https://github.com/rust-lang/rust/issues/6117
//@ run-pass
#![allow(dead_code)]

enum Either<T, U> { Left(T), Right(U) }

pub fn main() {
    match Either::Left(Box::new(17)) {
        Either::Right(()) => {}
        _ => {}
    }
}
