//! regression test for <https://github.com/rust-lang/rust/issues/17651>
// Test that moves of unsized values within closures are caught
// and rejected.

fn main() {
    (|| Box::new(*(&[0][..])))();
    //~^ ERROR the size for values of type
}
