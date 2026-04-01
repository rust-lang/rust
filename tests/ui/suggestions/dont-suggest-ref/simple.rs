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

    // move from Either/X place

    let X(_t) = *s;
    //~^ ERROR cannot move
    //~| HELP consider removing the dereference here
    if let Either::One(_t) = *r { }
    //~^ ERROR cannot move
    //~| HELP consider removing the dereference here
    while let Either::One(_t) = *r { }
    //~^ ERROR cannot move
    //~| HELP consider removing the dereference here
    match *r {
        //~^ ERROR cannot move
        //~| HELP consider removing the dereference here
        Either::One(_t)
        | Either::Two(_t) => (),
    }
    match *r {
        //~^ ERROR cannot move
        //~| HELP consider removing the dereference here
        Either::One(_t) => (),
        Either::Two(ref _t) => (),
        // FIXME: should suggest removing `ref` too
    }

    let X(_t) = *sm;
    //~^ ERROR cannot move
    //~| HELP consider removing the dereference here
    if let Either::One(_t) = *rm { }
    //~^ ERROR cannot move
    //~| HELP consider removing the dereference here
    while let Either::One(_t) = *rm { }
    //~^ ERROR cannot move
    //~| HELP consider removing the dereference here
    match *rm {
        //~^ ERROR cannot move
        //~| HELP consider removing the dereference here
        Either::One(_t)
        | Either::Two(_t) => (),
    }
    match *rm {
        //~^ ERROR cannot move
        //~| HELP consider removing the dereference here
        Either::One(_t) => (),
        Either::Two(ref _t) => (),
        // FIXME: should suggest removing `ref` too
    }
    match *rm {
        //~^ ERROR cannot move
        //~| HELP consider removing the dereference here
        Either::One(_t) => (),
        Either::Two(ref mut _t) => (),
        // FIXME: should suggest removing `ref` too
    }

    let X(_t) = vs[0];
    //~^ ERROR cannot move
    //~| HELP consider borrowing here
    if let Either::One(_t) = vr[0] { }
    //~^ ERROR cannot move
    //~| HELP consider borrowing here
    while let Either::One(_t) = vr[0] { }
    //~^ ERROR cannot move
    //~| HELP consider borrowing here
    match vr[0] {
        //~^ ERROR cannot move
        //~| HELP consider borrowing here
        Either::One(_t)
        | Either::Two(_t) => (),
    }
    match vr[0] {
        //~^ ERROR cannot move
        //~| HELP consider borrowing here
        Either::One(_t) => (),
        Either::Two(ref _t) => (),
        // FIXME: should suggest removing `ref` too
    }

    let X(_t) = vsm[0];
    //~^ ERROR cannot move
    //~| HELP consider borrowing here
    if let Either::One(_t) = vrm[0] { }
    //~^ ERROR cannot move
    //~| HELP consider borrowing here
    while let Either::One(_t) = vrm[0] { }
    //~^ ERROR cannot move
    //~| HELP consider borrowing here
    match vrm[0] {
        //~^ ERROR cannot move
        //~| HELP consider borrowing here
        Either::One(_t)
        | Either::Two(_t) => (),
    }
    match vrm[0] {
        //~^ ERROR cannot move
        //~| HELP consider borrowing here
        Either::One(_t) => (),
        Either::Two(ref _t) => (),
        // FIXME: should suggest removing `ref` too
    }
    match vrm[0] {
        //~^ ERROR cannot move
        //~| HELP consider borrowing here
        Either::One(_t) => (),
        Either::Two(ref mut _t) => (),
        // FIXME: should suggest removing `ref` too
    }

    // move from &Either/&X place

    let &X(_t) = s;
    //~^ ERROR cannot move
    //~| HELP consider removing
    if let &Either::One(_t) = r { }
    //~^ ERROR cannot move
    //~| HELP consider removing
    while let &Either::One(_t) = r { }
    //~^ ERROR cannot move
    //~| HELP consider removing
    match r {
        //~^ ERROR cannot move
        &Either::One(_t)
        //~^ HELP consider removing
        | &Either::Two(_t) => (),
        // FIXME: would really like a suggestion here too
    }
    match r {
        //~^ ERROR cannot move
        &Either::One(_t) => (),
        //~^ HELP consider removing
        &Either::Two(ref _t) => (),
    }
    match r {
        //~^ ERROR cannot move
        &Either::One(_t) => (),
        //~^ HELP consider removing
        Either::Two(_t) => (),
    }
    fn f1(&X(_t): &X) { }
    //~^ ERROR cannot move
    //~| HELP consider removing

    let &mut X(_t) = sm;
    //~^ ERROR cannot move
    //~| HELP consider removing
    if let &mut Either::One(_t) = rm { }
    //~^ ERROR cannot move
    //~| HELP consider removing
    while let &mut Either::One(_t) = rm { }
    //~^ ERROR cannot move
    //~| HELP consider removing
    match rm {
        //~^ ERROR cannot move
        &mut Either::One(_t) => (),
        //~^ HELP consider removing
        &mut Either::Two(_t) => (),
        //~^ HELP consider removing
    }
    match rm {
        //~^ ERROR cannot move
        &mut Either::One(_t) => (),
        //~^ HELP consider removing
        &mut Either::Two(ref _t) => (),
    }
    match rm {
        //~^ ERROR cannot move
        &mut Either::One(_t) => (),
        //~^ HELP consider removing
        &mut Either::Two(ref mut _t) => (),
    }
    match rm {
        //~^ ERROR cannot move
        &mut Either::One(_t) => (),
        //~^ HELP consider removing
        Either::Two(_t) => (),
    }
    fn f2(&mut X(_t): &mut X) { }
    //~^ ERROR cannot move
    //~| HELP consider removing

    // move from tuple of &Either/&X

    // FIXME: These should have suggestions.

    let (&X(_t),) = (&x.clone(),);
    //~^ ERROR cannot move
    //~| HELP consider removing the borrow
    if let (&Either::One(_t),) = (&e.clone(),) { }
    //~^ ERROR cannot move
    //~| HELP consider removing the borrow
    while let (&Either::One(_t),) = (&e.clone(),) { }
    //~^ ERROR cannot move
    //~| HELP consider removing the borrow
    match (&e.clone(),) {
        //~^ ERROR cannot move
        (&Either::One(_t),)
        //~^ HELP consider removing the borrow
        | (&Either::Two(_t),) => (),
    }
    fn f3((&X(_t),): (&X,)) { }
    //~^ ERROR cannot move
    //~| HELP consider removing the borrow

    let (&mut X(_t),) = (&mut xm.clone(),);
    //~^ ERROR cannot move
    //~| HELP consider removing the mutable borrow
    if let (&mut Either::One(_t),) = (&mut em.clone(),) { }
    //~^ ERROR cannot move
    //~| HELP consider removing the mutable borrow
    while let (&mut Either::One(_t),) = (&mut em.clone(),) { }
    //~^ ERROR cannot move
    //~| HELP consider removing the mutable borrow
    match (&mut em.clone(),) {
        //~^ ERROR cannot move
        (&mut Either::One(_t),) => (),
        //~^ HELP consider removing the mutable borrow
        (&mut Either::Two(_t),) => (),
        //~^ HELP consider removing the mutable borrow
    }
    fn f4((&mut X(_t),): (&mut X,)) { }
    //~^ ERROR cannot move
    //~| HELP consider removing the mutable borrow

    // move from &Either/&X value

    let &X(_t) = &x;
    //~^ ERROR cannot move
    //~| HELP consider removing
    if let &Either::One(_t) = &e { }
    //~^ ERROR cannot move
    //~| HELP consider removing
    while let &Either::One(_t) = &e { }
    //~^ ERROR cannot move
    //~| HELP consider removing
    match &e {
        //~^ ERROR cannot move
        &Either::One(_t)
        //~^ HELP consider removing
        | &Either::Two(_t) => (),
        // FIXME: would really like a suggestion here too
    }
    match &e {
        //~^ ERROR cannot move
        &Either::One(_t) => (),
        //~^ HELP consider removing
        &Either::Two(ref _t) => (),
    }
    match &e {
        //~^ ERROR cannot move
        &Either::One(_t) => (),
        //~^ HELP consider removing
        Either::Two(_t) => (),
    }

    let &mut X(_t) = &mut xm;
    //~^ ERROR cannot move
    //~| HELP consider removing
    if let &mut Either::One(_t) = &mut em { }
    //~^ ERROR cannot move
    //~| HELP consider removing
    while let &mut Either::One(_t) = &mut em { }
    //~^ ERROR cannot move
    //~| HELP consider removing
    match &mut em {
        //~^ ERROR cannot move
        &mut Either::One(_t)
        //~^ HELP consider removing
        | &mut Either::Two(_t) => (),
        // FIXME: would really like a suggestion here too
    }
    match &mut em {
        //~^ ERROR cannot move
        &mut Either::One(_t) => (),
        //~^ HELP consider removing
        &mut Either::Two(ref _t) => (),
    }
    match &mut em {
        //~^ ERROR cannot move
        &mut Either::One(_t) => (),
        //~^ HELP consider removing
        &mut Either::Two(ref mut _t) => (),
    }
    match &mut em {
        //~^ ERROR cannot move
        &mut Either::One(_t) => (),
        //~^ HELP consider removing
        Either::Two(_t) => (),
    }
}

struct Testing {
    a: Option<String>
}

fn testing(a: &Testing) {
    let Some(_s) = a.a else {
        //~^ ERROR cannot move
        //~| HELP consider borrowing the pattern binding
        return;
    };
}
