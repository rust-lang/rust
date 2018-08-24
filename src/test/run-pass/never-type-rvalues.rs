#![feature(never_type)]
#![allow(dead_code)]
#![allow(path_statements)]
#![allow(unreachable_patterns)]

fn never_direct(x: !) {
    x;
}

fn never_ref_pat(ref x: !) {
    *x;
}

fn never_ref(x: &!) {
    let &y = x;
    y;
}

fn never_pointer(x: *const !) {
    unsafe {
        *x;
    }
}

fn never_slice(x: &[!]) {
    x[0];
}

fn never_match(x: Result<(), !>) {
    match x {
        Ok(_) => {},
        Err(_) => {},
    }
}

pub fn main() { }
