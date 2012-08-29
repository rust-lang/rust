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

fn alloc(_bcx : &arena) -> &bcx unsafe {
    return unsafe::reinterpret_cast(
        &libc::malloc(sys::size_of::<bcx/&blk>() as libc::size_t));
}

fn h(bcx : &bcx) -> &bcx {
    return alloc(bcx.fcx.arena);
}

fn g(fcx : &fcx) {
    let bcx = { fcx: fcx };
    let bcx2 = h(&bcx);
    unsafe {
        libc::free(unsafe::reinterpret_cast(&bcx2));
    }
}

fn f(ccx : &ccx) {
    let a = arena(());
    let fcx = { arena: &a, ccx: ccx };
    return g(&fcx);
}

fn main() {
    let ccx = { x: 0 };
    f(&ccx);
}

