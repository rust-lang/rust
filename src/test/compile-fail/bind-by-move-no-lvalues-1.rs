struct X { x: (); drop { error!("destructor runs"); } }

fn main() {
    let x = some(X { x: () });
    match x {
        some(move _z) => { }, //~ ERROR cannot bind by-move when matching an lvalue
        none => fail
    }
}
