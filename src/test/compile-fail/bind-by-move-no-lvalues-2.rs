struct X { x: (); drop { error!("destructor runs"); } }
struct Y { y: option<X>; }

fn main() {
    let x = Y { y: some(X { x: () }) };
    match x.y {
        some(move _z) => { }, //~ ERROR cannot bind by-move when matching an lvalue
        none => fail
    }
}
