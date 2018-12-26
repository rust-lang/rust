struct X { x: (), }

impl Drop for X {
    fn drop(&mut self) {
        println!("destructor runs");
    }
}

fn main() {
    let x = Some((X { x: () }, X { x: () }));
    match x {
        Some((ref _y, _z)) => { }, //~ ERROR cannot bind by-move and by-ref in the same pattern
        None => panic!()
    }
}
