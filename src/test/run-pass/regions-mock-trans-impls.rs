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

impl arena for arena {
    fn alloc_inner(sz: uint, _align: uint) -> *() unsafe {
        ret unsafe::reinterpret_cast(libc::malloc(sz as libc::size_t));
    }
    fn alloc(tydesc: *()) -> *() {
        unsafe {
            let tydesc = tydesc as *sys::type_desc;
            self.alloc_inner((*tydesc).size, (*tydesc).align)
        }
    }
}

fn h(bcx : &bcx) -> &bcx {
    ret new(*bcx.fcx.arena) { fcx: bcx.fcx };
}

fn g(fcx : &fcx) {
    let bcx = { fcx: fcx };
    let bcx2 = h(&bcx);
    unsafe {
        libc::free(unsafe::reinterpret_cast(bcx2));
    }
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

