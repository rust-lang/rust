#![allow(dead_code)]
fn main() {
    // Original borrow ends at end of function
    let mut x = 1;
    let y = &mut x;
    //~^ NOTE_NONVIRAL mutable borrow occurs here
    let z = &x; //~ ERROR cannot borrow
    //~^ NOTE_NONVIRAL immutable borrow occurs here
    z.use_ref();
    y.use_mut();
}

fn foo() {
    match true {
        true => {
            // Original borrow ends at end of match arm
            let mut x = 1;
            let y = &x;
            //~^ NOTE_NONVIRAL immutable borrow occurs here
            let z = &mut x; //~ ERROR cannot borrow
            //~^ NOTE_NONVIRAL mutable borrow occurs here
            z.use_mut();
            y.use_ref();
        }
        false => ()
    }
}

fn bar() {
    // Original borrow ends at end of closure
    || {
        let mut x = 1;
        let y = &mut x;
        //~^ NOTE_NONVIRAL first mutable borrow occurs here
        let z = &mut x; //~ ERROR cannot borrow
        //~^ NOTE_NONVIRAL second mutable borrow occurs here
        z.use_mut();
        y.use_mut();
    };
}

trait Fake { fn use_mut(&mut self) { } fn use_ref(&self) { }  }
impl<T> Fake for T { }
