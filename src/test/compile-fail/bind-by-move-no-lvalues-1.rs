struct X { x: (), }

impl X : Drop {
    fn finalize(&self) {
        error!("destructor runs");
    }
}

fn main() {
    let x = Some(X { x: () });
    match x {
        Some(move _z) => { }, //~ ERROR cannot bind by-move when matching an lvalue
        None => fail
    }
}
