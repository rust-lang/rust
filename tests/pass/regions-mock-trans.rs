use std::mem;

struct Arena(());

struct Bcx<'a> {
    fcx: &'a Fcx<'a>,
}

#[allow(dead_code)]
struct Fcx<'a> {
    arena: &'a Arena,
    ccx: &'a Ccx,
}

#[allow(dead_code)]
struct Ccx {
    x: isize,
}

fn alloc<'a>(_bcx: &'a Arena) -> &'a mut Bcx<'a> {
    unsafe { mem::transmute(libc::malloc(mem::size_of::<Bcx<'a>>() as libc::size_t)) }
}

fn h<'a>(bcx: &'a Bcx<'a>) -> &'a mut Bcx<'a> {
    return alloc(bcx.fcx.arena);
}

fn g(fcx: &Fcx) {
    let bcx = Bcx { fcx: fcx };
    let bcx2 = h(&bcx);
    unsafe {
        libc::free(mem::transmute(bcx2));
    }
}

fn f(ccx: &Ccx) {
    let a = Arena(());
    let fcx = Fcx { arena: &a, ccx: ccx };
    return g(&fcx);
}

pub fn main() {
    let ccx = Ccx { x: 0 };
    f(&ccx);
}
