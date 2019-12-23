// This test is taken directly from #16053.
// It checks that you cannot use an AND-pattern (`binding @ pat`)
// where one side is by-ref and the other is by-move.

#![feature(bindings_after_at)]

struct X { x: () }

fn main() {
    let x = Some(X { x: () });
    match x {
        Some(ref _y @ _z) => { }, //~ ERROR cannot bind by-move and by-ref in the same pattern
        None => panic!()
    }

    let x = Some(X { x: () });
    match x {
        Some(_z @ ref _y) => { }, //~ ERROR cannot bind by-move with sub-bindings
        //~^ ERROR borrow of moved value
        None => panic!()
    }

    let mut x = Some(X { x: () });
    match x {
        Some(ref mut _y @ _z) => { }, //~ ERROR cannot bind by-move and by-ref in the same pattern
        None => panic!()
    }

    let mut x = Some(X { x: () });
    match x {
        Some(_z @ ref mut _y) => { }, //~ ERROR cannot bind by-move with sub-bindings
        //~^ ERROR borrow of moved value
        None => panic!()
    }
}
