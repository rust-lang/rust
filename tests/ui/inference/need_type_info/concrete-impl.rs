trait Ambiguous<A> {
    fn method() {}
}

struct One;
struct Two;
struct Struct;

impl Ambiguous<One> for Struct {}
//~^ NOTE multiple `impl`s satisfying `Struct: Ambiguous<_>` found
impl Ambiguous<Two> for Struct {}

fn main() {
    <Struct as Ambiguous<_>>::method();
    //~^ ERROR type annotations needed
    //~| NOTE cannot infer type of the type parameter `A`
}
