// run-pass
fn main() {
    fn test() -> Box<std::any::Any + 'static> { Box::new(1) }
    println!("{:?}", test())
}
