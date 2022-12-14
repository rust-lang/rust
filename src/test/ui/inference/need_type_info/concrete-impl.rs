trait Ambiguous<A> {
    fn method() {}
}

struct One;
struct Two;
struct Struct;

impl Ambiguous<One> for Struct {}
impl Ambiguous<Two> for Struct {}

fn main() {
    <Struct as Ambiguous<_>>::method();
    //~^ ERROR type annotations needed
    //~| ERROR type annotations needed
}
