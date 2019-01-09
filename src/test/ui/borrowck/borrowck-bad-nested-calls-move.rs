// Test that we detect nested calls that could free pointers evaluated
// for earlier arguments.

#![feature(box_syntax)]

fn rewrite(v: &mut Box<usize>) -> usize {
    *v = box 22;
    **v
}

fn add(v: &usize, w: Box<usize>) -> usize {
    *v + *w
}

fn implicit() {
    let mut a: Box<_> = box 1;

    // Note the danger here:
    //
    //    the pointer for the first argument has already been
    //    evaluated, but it gets moved when evaluating the second
    //    argument!
    add(
        &*a,
        a); //~ ERROR cannot move
}

fn explicit() {
    let mut a: Box<_> = box 1;
    add(
        &*a,
        a); //~ ERROR cannot move
}

fn main() {}
