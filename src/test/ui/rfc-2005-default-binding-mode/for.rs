struct Foo {}

pub fn main() {
    let mut tups = vec![(Foo{}, Foo{})];
    // The below desugars to &(ref n, mut m).
    for (n, mut m) in &tups {
        //~^ ERROR cannot bind by-move and by-ref in the same pattern
        //~| ERROR cannot move out of a shared reference
    }
}
