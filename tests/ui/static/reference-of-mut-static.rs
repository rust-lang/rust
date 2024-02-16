//@ revisions: e2021 e2024

//@ [e2021] edition:2021
//@ [e2024] compile-flags: --edition 2024 -Z unstable-options

#![deny(static_mut_ref)]

use std::ptr::{addr_of, addr_of_mut};

fn main() {
    static mut X: i32 = 1;

    static mut Y: i32 = 1;

    unsafe {
        let _y = &X;
        //[e2024]~^ ERROR reference of mutable static is disallowed
        //[e2021]~^^ ERROR shared reference of mutable static is discouraged [static_mut_ref]

        let _y = &mut X;
        //[e2024]~^ ERROR reference of mutable static is disallowed
        //[e2021]~^^ ERROR mutable reference of mutable static is discouraged [static_mut_ref]

        let _z = addr_of_mut!(X);

        let _p = addr_of!(X);

        let ref _a = X;
        //[e2024]~^ ERROR reference of mutable static is disallowed
        //[e2021]~^^ ERROR shared reference of mutable static is discouraged [static_mut_ref]

        let (_b, _c) = (&X, &Y);
        //[e2024]~^ ERROR reference of mutable static is disallowed
        //[e2021]~^^ ERROR shared reference of mutable static is discouraged [static_mut_ref]
        //[e2024]~^^^ ERROR reference of mutable static is disallowed
        //[e2021]~^^^^ ERROR shared reference of mutable static is discouraged [static_mut_ref]

        foo(&X);
        //[e2024]~^ ERROR reference of mutable static is disallowed
        //[e2021]~^^ ERROR shared reference of mutable static is discouraged [static_mut_ref]

        static mut Z: &[i32; 3] = &[0, 1, 2];

        let _ = Z.len();
        let _ = Z[0];
        let _ = format!("{:?}", Z);
    }
}

fn foo<'a>(_x: &'a i32) {}
