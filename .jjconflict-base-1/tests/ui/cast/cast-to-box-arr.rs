//! regression test for <https://github.com/rust-lang/rust/issues/22403>
//@ run-pass
fn main() {
    let x = Box::new([1, 2, 3]);
    let y = x as Box<[i32]>;
    println!("y: {:?}", y);
}
