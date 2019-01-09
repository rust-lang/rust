#![feature(box_patterns)]
#![feature(box_syntax)]
#![feature(slice_patterns)]

fn a() {
    let mut vec = [box 1, box 2, box 3];
    match vec {
        [box ref _a, _, _] => {
        //~^ borrow of `vec[..]` occurs here
            vec[0] = box 4; //~ ERROR cannot assign
            //~^ assignment to borrowed `vec[..]` occurs here
            _a.use_ref();
        }
    }
}

fn b() {
    let mut vec = vec![box 1, box 2, box 3];
    let vec: &mut [Box<isize>] = &mut vec;
    match vec {
        &mut [ref _b..] => {
        //~^ borrow of `vec[..]` occurs here
            vec[0] = box 4; //~ ERROR cannot assign
            //~^ assignment to borrowed `vec[..]` occurs here
            _b.use_ref();
        }
    }
}

fn c() {
    let mut vec = vec![box 1, box 2, box 3];
    let vec: &mut [Box<isize>] = &mut vec;
    match vec {
        &mut [_a, //~ ERROR cannot move out
            //~| cannot move out
            //~| to prevent move
            ..
        ] => {
            // Note: `_a` is *moved* here, but `b` is borrowing,
            // hence illegal.
            //
            // See comment in middle/borrowck/gather_loans/mod.rs
            // in the case covering these sorts of vectors.
        }
        _ => {}
    }
    let a = vec[0]; //~ ERROR cannot move out
    //~| cannot move out of here
}

fn d() {
    let mut vec = vec![box 1, box 2, box 3];
    let vec: &mut [Box<isize>] = &mut vec;
    match vec {
        &mut [ //~ ERROR cannot move out
        //~^ cannot move out
         _b] => {}
        _ => {}
    }
    let a = vec[0]; //~ ERROR cannot move out
    //~| cannot move out of here
}

fn e() {
    let mut vec = vec![box 1, box 2, box 3];
    let vec: &mut [Box<isize>] = &mut vec;
    match vec {
        &mut [_a, _b, _c] => {}  //~ ERROR cannot move out
        //~| cannot move out
        _ => {}
    }
    let a = vec[0]; //~ ERROR cannot move out
    //~| cannot move out of here
}

fn main() {}

trait Fake { fn use_mut(&mut self) { } fn use_ref(&self) { }  }
impl<T> Fake for T { }
