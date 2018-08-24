trait Trait {
    type Bar;
}

type Foo = Trait<F=i32>; //~ ERROR E0220
                         //~| ERROR E0191
fn main() {
}
