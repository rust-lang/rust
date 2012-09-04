use std;
import libc, sys, unsafe;
import std::arena::Arena;

type bcx = {
    fcx: &fcx
};

type fcx = {
    arena: &Arena,
    ccx: &ccx
};

type ccx = {
    x: int
};

fn h(bcx : &r/bcx) -> &r/bcx {
    return bcx.fcx.arena.alloc(|| { fcx: bcx.fcx });
}

fn g(fcx : &fcx) {
    let bcx = { fcx: fcx };
    h(&bcx);
}

fn f(ccx : &ccx) {
    let a = Arena();
    let fcx = &{ arena: &a, ccx: ccx };
    return g(fcx);
}

fn main() {
    let ccx = { x: 0 };
    f(&ccx);
}

