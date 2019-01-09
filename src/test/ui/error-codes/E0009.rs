fn main() {
    struct X { x: (), }
    let x = Some((X { x: () }, X { x: () }));
    match x {
        Some((y, ref z)) => {},
        //~^ ERROR E0009
        None => panic!()
    }
}
