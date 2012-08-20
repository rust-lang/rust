struct X { x: (); drop { error!("destructor runs"); } }
struct Y { y: Option<X>; }

fn main() {
    let x = Y { y: Some(X { x: () }) };
    match x.y {
        Some(move _z) => { }, //~ ERROR cannot bind by-move when matching an lvalue
        None => fail
    }
}
