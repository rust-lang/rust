//! Regression test for https://github.com/rust-lang/rust/issues/12744

//@ check-pass
fn main() {
    fn test() -> Box<dyn std::any::Any + 'static> { Box::new(1) }
    println!("{:?}", test())
}
