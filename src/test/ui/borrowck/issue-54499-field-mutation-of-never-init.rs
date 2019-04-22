#![warn(unused)]
#[derive(Debug)]
struct S(i32);

type Tuple = (S, i32);
struct Tpair(S, i32);
struct Spair { x: S, y: i32 }

fn main() {
    {
        let t: Tuple;
        t.0 = S(1);
        //~^ ERROR assign to part of possibly uninitialized variable: `t` [E0381]
        t.1 = 2;
        println!("{:?} {:?}", t.0, t.1);
    }

    {
        let u: Tpair;
        u.0 = S(1);
        //~^ ERROR assign to part of possibly uninitialized variable: `u` [E0381]
        u.1 = 2;
        println!("{:?} {:?}", u.0, u.1);
    }

    {
        let v: Spair;
        v.x = S(1);
        //~^ ERROR assign to part of possibly uninitialized variable: `v` [E0381]
        v.y = 2;
        println!("{:?} {:?}", v.x, v.y);
    }
}
