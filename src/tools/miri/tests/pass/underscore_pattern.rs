// Various tests ensuring that underscore patterns really just construct the place, but don't check its contents.
#![feature(never_type)]

use std::ptr;

fn main() {
    dangling_match();
    invalid_match();
    dangling_let();
    invalid_let();
    dangling_let_type_annotation();
    invalid_let_type_annotation();
    never();
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

    unsafe {
        let x: Uninit<!> = Uninit { uninit: () };
        match x.value {
            _ => {}
        }
    }
}

fn dangling_let() {
    unsafe {
        let ptr = ptr::without_provenance::<bool>(0x40);
        let _ = *ptr;
    }

    unsafe {
        let ptr = ptr::without_provenance::<!>(0x40);
        let _ = *ptr;
    }
}

fn invalid_let() {
    unsafe {
        let val = 3u8;
        let ptr = ptr::addr_of!(val).cast::<bool>();
        let _ = *ptr;
    }

    unsafe {
        let val = 3u8;
        let ptr = ptr::addr_of!(val).cast::<!>();
        let _ = *ptr;
    }
}

// Adding a type annotation used to change how MIR is generated, make sure we cover both cases.
fn dangling_let_type_annotation() {
    unsafe {
        let ptr = ptr::without_provenance::<bool>(0x40);
        let _: bool = *ptr;
    }

    unsafe {
        let ptr = ptr::without_provenance::<!>(0x40);
        let _: ! = *ptr;
    }
}

fn invalid_let_type_annotation() {
    unsafe {
        let val = 3u8;
        let ptr = ptr::addr_of!(val).cast::<bool>();
        let _: bool = *ptr;
    }

    unsafe {
        let val = 3u8;
        let ptr = ptr::addr_of!(val).cast::<!>();
        let _: ! = *ptr;
    }
}

// Regression test from <https://github.com/rust-lang/rust/issues/117288>.
fn never() {
    unsafe {
        let x = 3u8;
        let x: *const ! = &x as *const u8 as *const _;
        let _: ! = *x;
    }

    // Without a type annotation, make sure we don't implicitly coerce `!` to `()`
    // when we do the noop `*x` (as that would require a `!` *value*, creating
    // which is UB).
    unsafe {
        let x = 3u8;
        let x: *const ! = &x as *const u8 as *const _;
        let _ = *x;
    }
}
