use self::A::*;
use V; //~ ERROR `V` is ambiguous
use self::B::*;
enum A {
    V
}
enum B {
    V
}

fn main() {}
