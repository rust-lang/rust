#[cfg(doc)] //~ NOTE the item is gated here
pub struct Foo; //~ NOTE found an item that was configured out

fn main() {
    let f = Foo; //~ ERROR cannot find value `Foo` in this scope
    //~^ NOTE not found in this scope
}
