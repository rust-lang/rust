// run-pass
#![allow(dead_code)]
// pretty-expanded FIXME #23616

enum Either<T, U> { Left(T), Right(U) }

pub fn main() {
    match Either::Left(Box::new(17)) {
        Either::Right(()) => {}
        _ => {}
    }
}
