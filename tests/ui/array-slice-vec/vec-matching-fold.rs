//@ run-pass

use std::fmt::Debug;

fn foldl<T, U, F>(values: &[T],
                  initial: U,
                  mut function: F)
                  -> U where
    U: Clone+Debug, T:Debug,
    F: FnMut(U, &T) -> U,
{    match values {
        &[ref head, ref tail @ ..] =>
            foldl(tail, function(initial, head), function),
        &[] => {
            // FIXME: call guards
            let res = initial.clone(); res
        }
    }
}

fn foldr<T, U, F>(values: &[T],
                  initial: U,
                  mut function: F)
                  -> U where
    U: Clone,
    F: FnMut(&T, U) -> U,
{
    match values {
        &[ref head @ .., ref tail] =>
            foldr(head, function(tail, initial), function),
        &[] => {
            // FIXME: call guards
            let res = initial.clone(); res
        }
    }
}

pub fn main() {
    let x = &[1, 2, 3, 4, 5];

    let product = foldl(x, 1, |a, b| a * *b);
    assert_eq!(product, 120);

    let sum = foldr(x, 0, |a, b| *a + b);
    assert_eq!(sum, 15);
}
