#![feature(plugin, box_syntax)]
#![plugin(clippy)]
#![allow(warnings, clippy)]

#![deny(boxed_local)]

#[derive(Clone)]
struct A;

impl A {
    fn foo(&self){}
}

fn main() {
}

fn warn_call() {
    let x = box A; //~ ERROR local variable
    x.foo(); 
}

fn warn_arg(x: Box<A>) { //~ ERROR local variable
    x.foo();
}

fn warn_rename_call() {
    let x = box A;

    let y = x; //~ ERROR local variable
    y.foo(); // via autoderef
}

fn warn_notuse() {
    let bz = box A; //~ ERROR local variable
}

fn warn_pass() {
    let bz = box A; //~ ERROR local variable
    take_ref(&bz); // via deref coercion
}

fn nowarn_return() -> Box<A> {
    let fx = box A;
    fx // moved out, "escapes"
}

fn nowarn_move() {
    let bx = box A;
    drop(bx) // moved in, "escapes"
}
fn nowarn_call() {
    let bx = box A;
    bx.clone(); // method only available to Box, not via autoderef
}

fn nowarn_pass() {
    let bx = box A;
    take_box(&bx); // fn needs &Box
}


fn take_box(x: &Box<A>) {}
fn take_ref(x: &A) {}


fn nowarn_ref_take() {
    // false positive, should actually warn
    let x = box A; //~ ERROR local variable
    let y = &x;
    take_box(y);
}

fn nowarn_match() {
    let x = box A; // moved into a match
    match x {
        y => drop(y)
    }
}

fn warn_match() {
    let x = box A; //~ ERROR local variable
    match &x { // not moved
        ref y => ()
    }
}
