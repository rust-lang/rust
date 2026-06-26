//! Regression test for <https://github.com/rust-lang/rust/issues/26805>.
//@ run-pass

struct NonOrd;

fn main() {
    let _: Box<dyn Iterator<Item = _>> = Box::new(vec![NonOrd].into_iter());
}
