// Various tests ensuring that underscore patterns really just construct the place, but don't check its contents.
#![feature(strict_provenance)]
use std::ptr;

fn main() {
    dangling_match();
    invalid_match();
    dangling_let();
    invalid_let();
    dangling_let_type_annotation();
    invalid_let_type_annotation();
}

fn dangling_match() {
    let p = {
        let b = Box::new(42);
        &*b as *const i32
    };
    unsafe {
        match *p {
            _ => {}
        }
    }
}

fn invalid_match() {
    union Uninit<T: Copy> {
        value: T,
        uninit: (),
    }
    unsafe {
        let x: Uninit<bool> = Uninit { uninit: () };
        match x.value {
            _ => {}
        }
    }
}

fn dangling_let() {
    unsafe {
        let ptr = ptr::invalid::<bool>(0x40);
        let _ = *ptr;
    }
}

fn invalid_let() {
    unsafe {
        let val = 3u8;
        let ptr = ptr::addr_of!(val).cast::<bool>();
        let _ = *ptr;
    }
}

// Adding a type annotation used to change how MIR is generated, make sure we cover both cases.
fn dangling_let_type_annotation() {
    unsafe {
        let ptr = ptr::invalid::<bool>(0x40);
        let _: bool = *ptr;
    }
}

fn invalid_let_type_annotation() {
    unsafe {
        let val = 3u8;
        let ptr = ptr::addr_of!(val).cast::<bool>();
        let _: bool = *ptr;
    }
}

// FIXME: we should also test `!`, not just `bool` -- but that s currently buggy:
// https://github.com/rust-lang/rust/issues/117288
