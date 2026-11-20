//@ known-bug: #104685
//@ compile-flags: -Zextra-const-ub-checks
#![feature(extern_types)]

extern {
    pub type ExternType;
}

extern "C" {
    pub static EXTERN: ExternType;
}

pub static EMPTY: () = unsafe { &EXTERN; };

fn main() {}
