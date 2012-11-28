struct X { x: (), }

impl X : Drop {
    fn finalize(&self) {
        error!("destructor runs");
    }
}

fn main() {
    let x = Some((X { x: () }, X { x: () }));
    match move x {
        Some((move _y, ref _z)) => { }, //~ ERROR cannot bind by-move and by-ref in the same pattern
        None => fail
    }
}
