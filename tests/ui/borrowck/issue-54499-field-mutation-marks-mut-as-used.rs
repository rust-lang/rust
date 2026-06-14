//@ revisions: classic partial_init
//@[partial_init] check-pass

#![cfg_attr(partial_init, feature(partial_init_locals))]
#![warn(unused)]
#[derive(Debug)]
struct S(i32);

type Tuple = (S, i32);
struct Tpair(S, i32);
struct Spair {
    x: S,
    y: i32,
}

fn main() {
    {
        let mut t: Tuple;
        t.0 = S(1);
        //[classic]~^ ERROR E0381
        //[classic]~| ERROR E0658
        t.1 = 2;
        t.0 = S(2);
        println!("{:?} {:?}", t.0.0, t.1);
    }

    {
        let mut u: Tpair;
        u.0 = S(1);
        //[classic]~^ ERROR E0381
        //[classic]~| ERROR E0658
        u.1 = 2;
        println!("{:?} {:?}", u.0, u.1);
    }

    {
        let mut v: Spair;
        v.x = S(1);
        //[classic]~^ ERROR E0381
        //[classic]~| ERROR E0658
        v.y = 2;
        println!("{:?} {:?}", v.x, v.y);
    }
}
