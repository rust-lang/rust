//! Regression test for https://github.com/rust-lang/rust/issues/17897

//@ run-pass
fn action(mut cb: Box<dyn FnMut(usize) -> usize>) -> usize {
    cb(1)
}

pub fn main() {
    println!("num: {}", action(Box::new(move |u| u)));
}
