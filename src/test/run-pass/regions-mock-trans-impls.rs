use std;
import libc, sys, unsafe;
import std::arena::arena;

type bcx = {
    fcx: &fcx
};

type fcx = {
    arena: &arena,
    ccx: &ccx
};

type ccx = {
    x: int
};

fn h(bcx : &bcx) -> &bcx {
    return bcx.fcx.arena.alloc(|| { fcx: bcx.fcx });
}

fn g(fcx : &fcx) {
    let bcx = { fcx: fcx };
    h(&bcx);
}

fn f(ccx : &ccx) {
    let a = arena();
    let fcx = &{ arena: &a, ccx: ccx };
    return g(fcx);
}

fn main() {
    let ccx = { x: 0 };
    f(&ccx);
}

