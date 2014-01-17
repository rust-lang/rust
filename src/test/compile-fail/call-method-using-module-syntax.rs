struct T;

trait Test {
    fn hello() -> uint;
}

impl Test for T {
    fn hello() { 5 }
}

fn main() {
    T::hello();
    //~^ ERROR unresolved name
    //~^^ ERROR use of undeclared module `T`
    //~^^^ ERROR unresolved name `T::hello`
}
