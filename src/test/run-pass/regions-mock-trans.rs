import libc, sys, unsafe;

enum arena = ();

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

fn alloc(bcx : &a.arena) -> &a.bcx unsafe {
    ret unsafe::reinterpret_cast(libc::malloc(sys::size_of::<bcx>()));
}

fn h(bcx : &a.bcx) -> &a.bcx {
    ret alloc(bcx.fcx.arena);
}

fn g(fcx : &fcx) {
    let bcx = { fcx: fcx };
    let bcx2 = h(&bcx);
}

fn f(ccx : &ccx) {
    let a = arena(());
    let fcx = { arena: &a, ccx: ccx }; 
    ret g(&fcx);
}

fn main() {
    let ccx = { x: 0 };
    f(&ccx);
}

