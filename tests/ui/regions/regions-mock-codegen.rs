// run-pass
#![allow(dead_code)]
#![allow(non_camel_case_types)]
// pretty-expanded FIXME #23616
#![feature(allocator_api)]

use std::alloc::{handle_alloc_error, Allocator, Global, Layout};
use std::ptr::NonNull;

struct arena(());

struct Bcx<'a> {
    fcx: &'a Fcx<'a>,
}

struct Fcx<'a> {
    arena: &'a arena,
    ccx: &'a Ccx,
}

struct Ccx {
    x: isize,
}

fn allocate(_bcx: &arena) -> &Bcx<'_> {
    unsafe {
        let layout = Layout::new::<Bcx>();
        let ptr = Global.allocate(layout).unwrap_or_else(|_| handle_alloc_error(layout));
        &*(ptr.as_ptr() as *const _)
    }
}

fn h<'a>(bcx: &'a Bcx<'a>) -> &'a Bcx<'a> {
    return allocate(bcx.fcx.arena);
}

fn g(fcx: &Fcx) {
    let bcx = Bcx { fcx };
    let bcx2 = h(&bcx);
    unsafe {
        Global.deallocate(NonNull::new_unchecked(bcx2 as *const _ as *mut _), Layout::new::<Bcx>());
    }
}

fn f(ccx: &Ccx) {
    let a = arena(());
    let fcx = Fcx { arena: &a, ccx };
    return g(&fcx);
}

pub fn main() {
    let ccx = Ccx { x: 0 };
    f(&ccx);
}
