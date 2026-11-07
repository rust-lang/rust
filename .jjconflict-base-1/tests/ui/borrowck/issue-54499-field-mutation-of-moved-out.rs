#![warn(unused)]
#[derive(Debug)]
struct S(i32);

type Tuple = (S, i32);
struct Tpair(S, i32);
struct Spair { x: S, y: i32 }

fn main() {
    {
        let t: Tuple = (S(0), 0);
        drop(t);
        t.0 = S(1);
        //~^ ERROR assign to part of moved value: `t` [E0382]
        //~| ERROR cannot assign to `t.0`, as `t` is not declared as mutable [E0594]
        t.1 = 2;
        //~^ ERROR cannot assign to `t.1`, as `t` is not declared as mutable [E0594]
        println!("{:?} {:?}", t.0, t.1);
    }

    {
        let u: Tpair = Tpair(S(0), 0);
        drop(u);
        u.0 = S(1);
        //~^ ERROR assign to part of moved value: `u` [E0382]
        //~| ERROR cannot assign to `u.0`, as `u` is not declared as mutable [E0594]
        u.1 = 2;
        //~^ ERROR cannot assign to `u.1`, as `u` is not declared as mutable [E0594]
        println!("{:?} {:?}", u.0, u.1);
    }

    {
        let v: Spair = Spair { x: S(0), y: 0 };
        drop(v);
        v.x = S(1);
        //~^ ERROR assign to part of moved value: `v` [E0382]
        //~| ERROR cannot assign to `v.x`, as `v` is not declared as mutable [E0594]
        v.y = 2;
        //~^ ERROR cannot assign to `v.y`, as `v` is not declared as mutable [E0594]
        println!("{:?} {:?}", v.x, v.y);
    }
}
