#![feature(box_patterns)]


fn a() {
    let mut vec = [Box::new(1), Box::new(2), Box::new(3)];
    match vec {
        [box ref _a, _, _] => {
        //~^ NOTE borrow of `vec[_]` occurs here
            vec[0] = Box::new(4); //~ ERROR cannot assign
            //~^ NOTE assignment to borrowed `vec[_]` occurs here
            _a.use_ref();
            //~^ NOTE borrow later used here
        }
    }
}

fn b() {
    let mut vec = vec![Box::new(1), Box::new(2), Box::new(3)];
    let vec: &mut [Box<isize>] = &mut vec;
    match vec {
        &mut [ref _b @ ..] => {
        //~^ borrow of `vec[_]` occurs here
            vec[0] = Box::new(4); //~ ERROR cannot assign
            //~^ NOTE assignment to borrowed `vec[_]` occurs here
            _b.use_ref();
            //~^ NOTE borrow later used here
        }
    }
}

fn c() {
    let mut vec = vec![Box::new(1), Box::new(2), Box::new(3)];
    let vec: &mut [Box<isize>] = &mut vec;
    match vec {
        //~^ ERROR cannot move out
        //~| NOTE cannot move out
        &mut [_a,
        //~^ NOTE data moved here
        //~| NOTE move occurs because `_a` has type
        //~| HELP consider removing the `&mut`
            ..
        ] => {
        }
        _ => {}
    }
    let a = vec[0]; //~ ERROR cannot move out
    //~| NOTE cannot move out of here
    //~| NOTE move occurs because
    //~| HELP consider borrowing here
}

fn d() {
    let mut vec = vec![Box::new(1), Box::new(2), Box::new(3)];
    let vec: &mut [Box<isize>] = &mut vec;
    match vec {
        //~^ ERROR cannot move out
        //~| NOTE cannot move out
        &mut [
        //~^ HELP consider removing the `&mut`
         _b] => {}
        //~^ NOTE data moved here
        //~| NOTE move occurs because `_b` has type
        _ => {}
    }
    let a = vec[0]; //~ ERROR cannot move out
    //~| NOTE cannot move out of here
    //~| NOTE move occurs because
    //~| HELP consider borrowing here
}

fn e() {
    let mut vec = vec![Box::new(1), Box::new(2), Box::new(3)];
    let vec: &mut [Box<isize>] = &mut vec;
    match vec {
        //~^ ERROR cannot move out
        //~| NOTE cannot move out
        //~| NOTE move occurs because these variables have types
        &mut [_a, _b, _c] => {}
        //~^ NOTE data moved here
        //~| NOTE and here
        //~| NOTE and here
        //~| HELP consider removing the `&mut`
        _ => {}
    }
    let a = vec[0]; //~ ERROR cannot move out
    //~| NOTE cannot move out of here
    //~| NOTE move occurs because
    //~| HELP consider borrowing here
}

fn main() {}

trait Fake { fn use_mut(&mut self) { } fn use_ref(&self) { }  }
impl<T> Fake for T { }
