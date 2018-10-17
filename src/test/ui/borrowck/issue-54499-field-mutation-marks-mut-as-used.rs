// revisions: ast nll

// Since we are testing nll migration explicitly as a separate
// revision, don't worry about the --compare-mode=nll on this test.

// ignore-compare-mode-nll

//[ast]compile-flags: -Z borrowck=ast
//[nll]compile-flags: -Z borrowck=migrate -Z two-phase-borrows

#![warn(unused)]
#[derive(Debug)]
struct S(i32);

type Tuple = (S, i32);
struct Tpair(S, i32);
struct Spair { x: S, y: i32 }

fn main() {
    {
        let mut t: Tuple;
        t.0 = S(1);
        //[nll]~^ ERROR assign to part of possibly uninitialized variable: `t` [E0381]
        t.1 = 2;
        println!("{:?} {:?}", t.0, t.1);
        //[ast]~^ ERROR use of possibly uninitialized variable: `t.0` [E0381]
        //[ast]~| ERROR use of possibly uninitialized variable: `t.1` [E0381]
    }

    {
        let mut u: Tpair;
        u.0 = S(1);
        //[nll]~^ ERROR assign to part of possibly uninitialized variable: `u` [E0381]
        u.1 = 2;
        println!("{:?} {:?}", u.0, u.1);
        //[ast]~^ ERROR use of possibly uninitialized variable: `u.0` [E0381]
        //[ast]~| ERROR use of possibly uninitialized variable: `u.1` [E0381]
    }

    {
        let mut v: Spair;
        v.x = S(1);
        //[nll]~^ ERROR assign to part of possibly uninitialized variable: `v` [E0381]
        v.y = 2;
        println!("{:?} {:?}", v.x, v.y);
        //[ast]~^ ERROR use of possibly uninitialized variable: `v.x` [E0381]
        //[ast]~| ERROR use of possibly uninitialized variable: `v.y` [E0381]
    }
}
