struct X { x: (), }

impl X : Drop {
    fn finalize(&self) {
        error!("destructor runs");
    }
}

fn main() {
    let x = Some(X { x: () });
    match move x {
        Some(move _y @ ref _z) => { }, //~ ERROR cannot bind by-move with sub-bindings
        None => fail
    }
}
