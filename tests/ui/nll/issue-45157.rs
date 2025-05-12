#![allow(unused)]


#[derive(Clone, Copy, Default)]
struct S {
    a: u8,
    b: u8,
}
#[derive(Clone, Copy, Default)]
struct Z {
    c: u8,
    d: u8,
}

union U {
    s: S,
    z: Z,
}

fn main() {
    unsafe {
        let mut u = U { s: Default::default() };

        let mref = &mut u.s.a;
        *mref = 22;

        let nref = &u.z.c;
        //~^ ERROR cannot borrow `u` (via `u.z.c`) as immutable because it is also borrowed as mutable (via `u.s.a`) [E0502]
        println!("{} {}", mref, nref)
    }
}
