#[derive(Clone)]
enum Either {
    One(X),
    Two(X),
}

#[derive(Clone)]
struct X(Y);

#[derive(Clone)]
struct Y;

fn consume_fn<F: Fn()>(_f: F) { }
//~^ HELP `Fn` and `FnMut` closures
//~| HELP `Fn` and `FnMut` closures
//~| HELP `Fn` and `FnMut` closures
//~| HELP `Fn` and `FnMut` closures
//~| HELP `Fn` and `FnMut` closures
//~| HELP `Fn` and `FnMut` closures
//~| HELP `Fn` and `FnMut` closures
//~| HELP `Fn` and `FnMut` closures
//~| HELP `Fn` and `FnMut` closures
//~| HELP `Fn` and `FnMut` closures

fn consume_fnmut<F: FnMut()>(_f: F) { }
//~^ HELP `Fn` and `FnMut` closures
//~| HELP `Fn` and `FnMut` closures
//~| HELP `Fn` and `FnMut` closures
//~| HELP `Fn` and `FnMut` closures
//~| HELP `Fn` and `FnMut` closures
//~| HELP `Fn` and `FnMut` closures
//~| HELP `Fn` and `FnMut` closures
//~| HELP `Fn` and `FnMut` closures
//~| HELP `Fn` and `FnMut` closures
//~| HELP `Fn` and `FnMut` closures
//~| HELP `Fn` and `FnMut` closures

trait T {
    fn consume_fn<F: Fn()>(_f: F) { }
    //~^ HELP `Fn` and `FnMut` closures
    //~| HELP `Fn` and `FnMut` closures
    //~| HELP `Fn` and `FnMut` closures
    //~| HELP `Fn` and `FnMut` closures
    //~| HELP `Fn` and `FnMut` closures
    //~| HELP `Fn` and `FnMut` closures
    //~| HELP `Fn` and `FnMut` closures
    //~| HELP `Fn` and `FnMut` closures
    //~| HELP `Fn` and `FnMut` closures
    //~| HELP `Fn` and `FnMut` closures
    fn method_consume_fn<F: Fn()>(&self, _f: F) { }
    //~^ HELP `Fn` and `FnMut` closures
    //~| HELP `Fn` and `FnMut` closures
    //~| HELP `Fn` and `FnMut` closures
    //~| HELP `Fn` and `FnMut` closures
    //~| HELP `Fn` and `FnMut` closures
    //~| HELP `Fn` and `FnMut` closures
    //~| HELP `Fn` and `FnMut` closures
    //~| HELP `Fn` and `FnMut` closures
    //~| HELP `Fn` and `FnMut` closures
    //~| HELP `Fn` and `FnMut` closures
}
impl T for () {}

pub fn main() { }

fn move_into_fn() {
    let e = Either::One(X(Y));
    let mut em = Either::One(X(Y));

    let x = X(Y);

    // move into Fn

    consume_fn(|| {
        let X(_t) = x;
        //~^ ERROR cannot move
        //~| HELP consider borrowing here
        if let Either::One(_t) = e { }
        //~^ ERROR cannot move
        //~| HELP consider borrowing here
        while let Either::One(_t) = e { }
        //~^ ERROR cannot move
        //~| HELP consider borrowing here
        match e {
            //~^ ERROR cannot move
            //~| HELP consider borrowing here
            Either::One(_t)
            | Either::Two(_t) => (),
        }
        match e {
            //~^ ERROR cannot move
            //~| HELP consider borrowing here
            Either::One(_t) => (),
            Either::Two(ref _t) => (),
            // FIXME: should suggest removing `ref` too
        }

        let X(mut _t) = x;
        //~^ ERROR cannot move
        //~| HELP consider borrowing here
        if let Either::One(mut _t) = em { }
        //~^ ERROR cannot move
        //~| HELP consider borrowing here
        while let Either::One(mut _t) = em { }
        //~^ ERROR cannot move
        //~| HELP consider borrowing here
        match em {
            //~^ ERROR cannot move
            //~| HELP consider borrowing here
            Either::One(mut _t)
            | Either::Two(mut _t) => (),
        }
        match em {
            //~^ ERROR cannot move
            //~| HELP consider borrowing here
            Either::One(mut _t) => (),
            Either::Two(ref _t) => (),
            // FIXME: should suggest removing `ref` too
        }
    });
}

fn move_into_assoc_fn() {
    let e = Either::One(X(Y));
    let mut em = Either::One(X(Y));

    let x = X(Y);

    // move into Fn

    <() as T>::consume_fn(|| {
        let X(_t) = x;
        //~^ ERROR cannot move
        //~| HELP consider borrowing here
        if let Either::One(_t) = e { }
        //~^ ERROR cannot move
        //~| HELP consider borrowing here
        while let Either::One(_t) = e { }
        //~^ ERROR cannot move
        //~| HELP consider borrowing here
        match e {
            //~^ ERROR cannot move
            //~| HELP consider borrowing here
            Either::One(_t)
            | Either::Two(_t) => (),
        }
        match e {
            //~^ ERROR cannot move
            //~| HELP consider borrowing here
            Either::One(_t) => (),
            Either::Two(ref _t) => (),
            // FIXME: should suggest removing `ref` too
        }

        let X(mut _t) = x;
        //~^ ERROR cannot move
        //~| HELP consider borrowing here
        if let Either::One(mut _t) = em { }
        //~^ ERROR cannot move
        //~| HELP consider borrowing here
        while let Either::One(mut _t) = em { }
        //~^ ERROR cannot move
        //~| HELP consider borrowing here
        match em {
            //~^ ERROR cannot move
            //~| HELP consider borrowing here
            Either::One(mut _t)
            | Either::Two(mut _t) => (),
        }
        match em {
            //~^ ERROR cannot move
            //~| HELP consider borrowing here
            Either::One(mut _t) => (),
            Either::Two(ref _t) => (),
            // FIXME: should suggest removing `ref` too
        }
    });
}

fn move_into_method() {
    let e = Either::One(X(Y));
    let mut em = Either::One(X(Y));

    let x = X(Y);

    // move into Fn

    ().method_consume_fn(|| {
        let X(_t) = x;
        //~^ ERROR cannot move
        //~| HELP consider borrowing here
        if let Either::One(_t) = e { }
        //~^ ERROR cannot move
        //~| HELP consider borrowing here
        while let Either::One(_t) = e { }
        //~^ ERROR cannot move
        //~| HELP consider borrowing here
        match e {
            //~^ ERROR cannot move
            //~| HELP consider borrowing here
            Either::One(_t)
            | Either::Two(_t) => (),
        }
        match e {
            //~^ ERROR cannot move
            //~| HELP consider borrowing here
            Either::One(_t) => (),
            Either::Two(ref _t) => (),
            // FIXME: should suggest removing `ref` too
        }

        let X(mut _t) = x;
        //~^ ERROR cannot move
        //~| HELP consider borrowing here
        if let Either::One(mut _t) = em { }
        //~^ ERROR cannot move
        //~| HELP consider borrowing here
        while let Either::One(mut _t) = em { }
        //~^ ERROR cannot move
        //~| HELP consider borrowing here
        match em {
            //~^ ERROR cannot move
            //~| HELP consider borrowing here
            Either::One(mut _t)
            | Either::Two(mut _t) => (),
        }
        match em {
            //~^ ERROR cannot move
            //~| HELP consider borrowing here
            Either::One(mut _t) => (),
            Either::Two(ref _t) => (),
            // FIXME: should suggest removing `ref` too
        }
    });
}

fn move_into_fnmut() {
    let e = Either::One(X(Y));
    let mut em = Either::One(X(Y));

    let x = X(Y);

    // move into FnMut

    consume_fnmut(|| {
        let X(_t) = x;
        //~^ ERROR cannot move
        //~| HELP consider borrowing here
        if let Either::One(_t) = e { }
        //~^ ERROR cannot move
        //~| HELP consider borrowing here
        while let Either::One(_t) = e { }
        //~^ ERROR cannot move
        //~| HELP consider borrowing here
        match e {
            //~^ ERROR cannot move
            //~| HELP consider borrowing here
            Either::One(_t)
            | Either::Two(_t) => (),
        }
        match e {
            //~^ ERROR cannot move
            //~| HELP consider borrowing here
            Either::One(_t) => (),
            Either::Two(ref _t) => (),
            // FIXME: should suggest removing `ref` too
        }

        let X(mut _t) = x;
        //~^ ERROR cannot move
        //~| HELP consider borrowing here
        if let Either::One(mut _t) = em { }
        //~^ ERROR cannot move
        //~| HELP consider borrowing here
        while let Either::One(mut _t) = em { }
        //~^ ERROR cannot move
        //~| HELP consider borrowing here
        match em {
            //~^ ERROR cannot move
            //~| HELP consider borrowing here
            Either::One(mut _t)
            | Either::Two(mut _t) => (),
        }
        match em {
            //~^ ERROR cannot move
            //~| HELP consider borrowing here
            Either::One(mut _t) => (),
            Either::Two(ref _t) => (),
            // FIXME: should suggest removing `ref` too
        }
        match em {
            //~^ ERROR cannot move
            //~| HELP consider borrowing here
            Either::One(mut _t) => (),
            Either::Two(ref mut _t) => (),
            // FIXME: should suggest removing `ref` too
        }
    });
}
