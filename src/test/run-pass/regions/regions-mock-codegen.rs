// run-pass
#![allow(dead_code)]
#![allow(non_camel_case_types)]

// pretty-expanded FIXME #23616

#![feature(allocator_api)]

use std::alloc::{Alloc, Global, Layout, handle_alloc_error};
use std::ptr::NonNull;

struct arena(());

struct Bcx<'a> {
    fcx: &'a Fcx<'a>
}

struct Fcx<'a> {
    arena: &'a arena,
    ccx: &'a Ccx
}

struct Ccx {
    x: isize
}

fn alloc<'a>(_bcx : &'a arena) -> &'a Bcx<'a> {
    unsafe {
        let layout = Layout::new::<Bcx>();
        let ptr = Global.alloc(layout).unwrap_or_else(|_| handle_alloc_error(layout));
        &*(ptr.as_ptr() as *const _)
    }
}

fn h<'a>(bcx : &'a Bcx<'a>) -> &'a Bcx<'a> {
    return alloc(bcx.fcx.arena);
}

fn g(fcx : &Fcx) {
    let bcx = Bcx { fcx: fcx };
    let bcx2 = h(&bcx);
    unsafe {
        Global.dealloc(NonNull::new_unchecked(bcx2 as *const _ as *mut _), Layout::new::<Bcx>());
    }
}

fn f(ccx : &Ccx) {
    let a = arena(());
    let fcx = Fcx { arena: &a, ccx: ccx };
    return g(&fcx);
}

pub fn main() {
    let ccx = Ccx { x: 0 };
    f(&ccx);
}
