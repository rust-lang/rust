//! Regression test for https://github.com/rust-lang/rust/issues/10436

//@ run-pass
fn works<T>(x: T) -> Vec<T> { vec![x] }

fn also_works<T: Clone>(x: T) -> Vec<T> { vec![x] }

fn main() {
    let _: Vec<usize> = works(0);
    let _: Vec<usize> = also_works(0);
    let _ = works(0);
    let _ = also_works(0);
}
