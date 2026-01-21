//! regression test for https://github.com/rust-lang/rust/issues/43205
//@ run-pass
fn main() {
    let _ = &&[()][0];
    println!("{:?}", &[(), ()][1]);
}
