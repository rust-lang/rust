// build-pass (FIXME(62277): could be check-pass?)

// ignore-tidy-linelength

// Issue #21633: reject duplicate loop labels in function bodies.
// This is testing the exact cases that are in the issue description.

#[allow(unused_labels)]
fn foo() {
    'fl: for _ in 0..10 { break; }
    'fl: loop { break; }           //~ WARN label name `'fl` shadows a label name that is already in scope

    'lf: loop { break; }
    'lf: for _ in 0..10 { break; } //~ WARN label name `'lf` shadows a label name that is already in scope
    'wl: while 2 > 1 { break; }
    'wl: loop { break; }           //~ WARN label name `'wl` shadows a label name that is already in scope
    'lw: loop { break; }
    'lw: while 2 > 1 { break; }    //~ WARN label name `'lw` shadows a label name that is already in scope
    'fw: for _ in 0..10 { break; }
    'fw: while 2 > 1 { break; }    //~ WARN label name `'fw` shadows a label name that is already in scope
    'wf: while 2 > 1 { break; }
    'wf: for _ in 0..10 { break; } //~ WARN label name `'wf` shadows a label name that is already in scope
    'tl: while let Some(_) = None::<i32> { break; }
    'tl: loop { break; }           //~ WARN label name `'tl` shadows a label name that is already in scope
    'lt: loop { break; }
    'lt: while let Some(_) = None::<i32> { break; }
                                   //~^ WARN label name `'lt` shadows a label name that is already in scope
}

// Note however that it is okay for the same label to be reused in
// different methods of one impl, as illustrated here.

struct S;
impl S {
    fn m1(&self) { 'okay: loop { break 'okay; } }
    fn m2(&self) { 'okay: loop { break 'okay; } }
}


pub fn main() {
    let s = S;
    s.m1();
    s.m2();
    foo();
}
