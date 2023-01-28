#[derive(Clone)]
enum Either {
    One(X),
    Two(X),
}

#[derive(Clone)]
struct X(Y);

#[derive(Clone)]
struct Y;


pub fn main() {
    let e = Either::One(X(Y));
    let mut em = Either::One(X(Y));

    let r = &e;
    let rm = &mut Either::One(X(Y));

    let x = X(Y);
    let mut xm = X(Y);

    let s = &x;
    let sm = &mut X(Y);

    let ve = vec![Either::One(X(Y))];

    let vr = &ve;
    let vrm = &mut vec![Either::One(X(Y))];

    let vx = vec![X(Y)];

    let vs = &vx;
    let vsm = &mut vec![X(Y)];

    // test for duplicate suggestions

    let &(X(_t), X(_u)) = &(x.clone(), x.clone());
    //~^ ERROR cannot move
    //~| HELP consider removing the borrow
    if let &(Either::One(_t), Either::Two(_u)) = &(e.clone(), e.clone()) { }
    //~^ ERROR cannot move
    //~| HELP consider removing the borrow
    while let &(Either::One(_t), Either::Two(_u)) = &(e.clone(), e.clone()) { }
    //~^ ERROR cannot move
    //~| HELP consider removing the borrow
    match &(e.clone(), e.clone()) {
        //~^ ERROR cannot move
        &(Either::One(_t), Either::Two(_u)) => (),
        //~^ HELP consider removing the borrow
        &(Either::Two(_t), Either::One(_u)) => (),
        //~^ HELP consider removing the borrow
        _ => (),
    }
    match &(e.clone(), e.clone()) {
        //~^ ERROR cannot move
        &(Either::One(_t), Either::Two(_u))
        //~^ HELP consider removing the borrow
        | &(Either::Two(_t), Either::One(_u)) => (),
        // FIXME: would really like a suggestion here too
        _ => (),
    }
    match &(e.clone(), e.clone()) {
        //~^ ERROR cannot move
        &(Either::One(_t), Either::Two(_u)) => (),
        //~^ HELP consider removing the borrow
        &(Either::Two(ref _t), Either::One(ref _u)) => (),
        _ => (),
    }
    match &(e.clone(), e.clone()) {
        //~^ ERROR cannot move
        &(Either::One(_t), Either::Two(_u)) => (),
        //~^ HELP consider removing the borrow
        (Either::Two(_t), Either::One(_u)) => (),
        _ => (),
    }
    fn f5(&(X(_t), X(_u)): &(X, X)) { }
    //~^ ERROR cannot move
    //~| HELP consider removing the borrow

    let &mut (X(_t), X(_u)) = &mut (xm.clone(), xm.clone());
    //~^ ERROR cannot move
    //~| HELP consider removing the mutable borrow
    if let &mut (Either::One(_t), Either::Two(_u)) = &mut (em.clone(), em.clone()) { }
    //~^ ERROR cannot move
    //~| HELP consider removing the mutable borrow
    while let &mut (Either::One(_t), Either::Two(_u)) = &mut (em.clone(), em.clone()) { }
    //~^ ERROR cannot move
    //~| HELP consider removing the mutable borrow
    match &mut (em.clone(), em.clone()) {
        //~^ ERROR cannot move
        &mut (Either::One(_t), Either::Two(_u)) => (),
        //~^ HELP consider removing the mutable borrow
        &mut (Either::Two(_t), Either::One(_u)) => (),
        //~^ HELP consider removing the mutable borrow
        _ => (),
    }
    match &mut (em.clone(), em.clone()) {
        //~^ ERROR cannot move
        &mut (Either::One(_t), Either::Two(_u))
        //~^ HELP consider removing the mutable borrow
        | &mut (Either::Two(_t), Either::One(_u)) => (),
        // FIXME: would really like a suggestion here too
        _ => (),
    }
    match &mut (em.clone(), em.clone()) {
        //~^ ERROR cannot move
        &mut (Either::One(_t), Either::Two(_u)) => (),
        //~^ HELP consider removing the mutable borrow
        &mut (Either::Two(ref _t), Either::One(ref _u)) => (),
        _ => (),
    }
    match &mut (em.clone(), em.clone()) {
        //~^ ERROR cannot move
        &mut (Either::One(_t), Either::Two(_u)) => (),
        //~^ HELP consider removing the mutable borrow
        &mut (Either::Two(ref mut _t), Either::One(ref mut _u)) => (),
        _ => (),
    }
    match &mut (em.clone(), em.clone()) {
        //~^ ERROR cannot move
        &mut (Either::One(_t), Either::Two(_u)) => (),
        //~^ HELP consider removing the mutable borrow
        (Either::Two(_t), Either::One(_u)) => (),
        _ => (),
    }
    fn f6(&mut (X(_t), X(_u)): &mut (X, X)) { }
    //~^ ERROR cannot move
    //~| HELP consider removing the mutable borrow
}
