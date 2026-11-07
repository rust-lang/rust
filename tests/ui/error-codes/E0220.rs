trait Trait {
    type Bar;
}

type Foo = dyn Trait<F=i32>; //~ ERROR E0220
fn main() {
}
