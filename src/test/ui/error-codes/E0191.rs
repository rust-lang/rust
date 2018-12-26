trait Trait {
    type Bar;
}

type Foo = Trait; //~ ERROR E0191

fn main() {}
