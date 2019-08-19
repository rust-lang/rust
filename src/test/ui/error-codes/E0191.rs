trait Trait {
    type Bar;
}

type Foo = dyn Trait; //~ ERROR E0191

fn main() {}
