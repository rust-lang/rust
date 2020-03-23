fn main() {
    #[derive(Clone)]
    struct X {
        x: (),
    }
    let mut tup = (X { x: () }, X { x: () });
    match Some(tup.clone()) {
        Some((y, ref z)) => {}
        //~^ ERROR binding by-move and by-ref in the same pattern is unstable
        None => panic!(),
    }

    let (ref a, b) = tup.clone();
    //~^ ERROR binding by-move and by-ref in the same pattern is unstable

    let (a, mut b) = &tup;
    //~^ ERROR binding by-move and by-ref in the same pattern is unstable
    //~| ERROR cannot move out of a shared reference

    let (mut a, b) = &mut tup;
    //~^ ERROR binding by-move and by-ref in the same pattern is unstable
    //~| ERROR cannot move out of a mutable reference
}
