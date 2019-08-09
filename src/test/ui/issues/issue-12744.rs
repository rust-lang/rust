// run-pass
fn main() {
    fn test() -> Box<dyn std::any::Any + 'static> { Box::new(1) }
    println!("{:?}", test())
}
