#![warn(unused)]
#[derive(Debug)]
struct S(i32);

type Tuple = (S, i32);
struct Tpair(S, i32);
struct Spair { x: S, y: i32 }

fn main() {
    {
        let mut t: Tuple = (S(0), 0);
        drop(t);
        t.0 = S(1);
        //~^ ERROR assign to part of moved value
        t.1 = 2;
        println!("{:?} {:?}", t.0, t.1);
    }

    {
        let mut u: Tpair = Tpair(S(0), 0);
        drop(u);
        u.0 = S(1);
        //~^ ERROR assign to part of moved value
        u.1 = 2;
        println!("{:?} {:?}", u.0, u.1);
    }

    {
        let mut v: Spair = Spair { x: S(0), y: 0 };
        drop(v);
        v.x = S(1);
        //~^ ERROR assign to part of moved value
        v.y = 2;
        println!("{:?} {:?}", v.x, v.y);
    }
}
