#![deny(unused_must_use)]

#[must_use]
struct S;

#[must_use]
trait A {}

struct B;

impl A for B {}

struct T(S);

struct U {
    x: (),
    y: T,
}

struct V {
    a: S,
}

struct W {
    w: [(u8, Box<dyn A>); 2],
    x: u32,
    y: (B, B),
    z: (S, S),
    e: [(u8, Box<dyn A>); 2],
    f: S,
}

fn get_v() -> V {
    V { a: S }
}

struct Z([(u8, Box<dyn A>); 2]);

fn get_wrapped_arr() -> Z {
    Z([(0, Box::new(B)), (0, Box::new(B))])
}

fn get_tuple_arr() -> ([(u8, Box<dyn A>); 2],) {
    ([(0, Box::new(B)), (0, Box::new(B))],)
}

struct R<T> {
    r: T
}

struct List<T>(T, Option<Box<Self>>);

fn main() {
    S; //~ ERROR unused `S` that must be used
    T(S); //~ ERROR unused `S` in field `0` that must be used
    U { x: (), y: T(S) }; //~ ERROR unused `S` in field `0` that must be used
    get_v(); //~ ERROR unused `S` in field `a` that must be used
    V { a: S }; //~ ERROR unused `S` in field `a` that must be used
    W {
        w: [(0, Box::new(B)), (0, Box::new(B))],
        //~^ ERROR unused array of boxed `A` trait objects in tuple element 1 that must be used
        x: 0,
        y: (B, B),
        z: (S, S),
        //~^ unused `S` in tuple element 0 that must be used
        //~^^ unused `S` in tuple element 1 that must be used
        e: [(0, Box::new(B)), (0, Box::new(B))],
        //~^ unused array of boxed `A` trait objects in tuple element 1 that must be used
        f: S, //~ ERROR unused `S` in field `f` that must be used
    };
    get_wrapped_arr();
    //~^ ERROR unused array of boxed `A` trait objects in tuple element 1 that must be use
    get_tuple_arr();
    //~^ ERROR unused array of boxed `A` trait objects in tuple element 1 that must be used
    R { r: S }; //~ ERROR unused `S` in field `r` that must be used
    List(S, Some(Box::new(List(S, None)))); //~ ERROR unused `S` in field `0` that must be used
}
