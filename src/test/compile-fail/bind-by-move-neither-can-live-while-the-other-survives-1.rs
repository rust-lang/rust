struct X { x: (); drop { error!("destructor runs"); } }

fn main() {
    let x = some(X { x: () });
    match move x {
        some(ref _y @ move _z) => { }, //~ ERROR cannot bind by-move and by-ref in the same pattern
        none => fail
    }
}
