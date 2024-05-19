#![feature(rustc_attrs)]
#![allow(dead_code)]
fn main() { #![rustc_error] // rust-lang/rust#49855
    // Original borrow ends at end of function
    let mut x = 1;
    let y = &mut x;
    //~^ mutable borrow occurs here
    let z = &x; //~ ERROR cannot borrow
    //~^ immutable borrow occurs here
    z.use_ref();
    y.use_mut();
}

fn foo() {
    match true {
        true => {
            // Original borrow ends at end of match arm
            let mut x = 1;
            let y = &x;
            //~^ immutable borrow occurs here
            let z = &mut x; //~ ERROR cannot borrow
            //~^ mutable borrow occurs here
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
        //~^ first mutable borrow occurs here
        let z = &mut x; //~ ERROR cannot borrow
        //~^ second mutable borrow occurs here
        z.use_mut();
        y.use_mut();
    };
}

trait Fake { fn use_mut(&mut self) { } fn use_ref(&self) { }  }
impl<T> Fake for T { }
