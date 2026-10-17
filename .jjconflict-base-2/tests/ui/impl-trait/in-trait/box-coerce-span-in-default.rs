//@ check-pass


struct TestA {}
struct TestB {}

impl TestTrait for TestA {
    type Output = ();
}
impl TestTrait for TestB {
    type Output = ();
}

trait TestTrait {
    type Output;
}

impl<A, B> TestTrait for GreeterOutput<A, B>
where
    A: TestTrait<Output = ()>,
    B: TestTrait<Output = ()>,
{
    type Output = ();
}

enum GreeterOutput<A, B>
where
    A: TestTrait<Output = ()>,
    B: TestTrait<Output = ()>,
{
    SayHello(A),
    SayGoodbye(B),
}

trait Greeter {
    fn test_func(&self, func: &str) -> impl TestTrait<Output = ()> {
        match func {
            "SayHello" => GreeterOutput::SayHello(TestA {}),
            "SayGoodbye" => GreeterOutput::SayGoodbye(TestB {}),
            _ => GreeterOutput::SayHello(TestA {}),
        }
    }
}

fn main() {
    println!("Hello, world!");
}
