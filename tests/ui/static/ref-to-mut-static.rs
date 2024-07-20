//@ revisions: e2021 e2024

//@ [e2021] edition:2021
//@ [e2024] compile-flags: --edition 2024 -Z unstable-options

#![deny(static_mut_refs)]

use std::ptr::{addr_of, addr_of_mut};

static mut FOO: (u32, u32) = (1, 2);

macro_rules! bar {
    ($x:expr) => {
        &mut ($x.0)
        //[e2024]~^ ERROR creating a mutable reference to a mutable static [E0796]
        //[e2021]~^^ ERROR creating a mutable reference to mutable static is discouraged [static_mut_refs]
    };
}

fn main() {
    static mut X: i32 = 1;

    static mut Y: i32 = 1;

    struct TheStruct {
        pub value: i32,
    }
    struct MyStruct {
        pub value: i32,
        pub s: TheStruct,
    }

    static mut A: MyStruct = MyStruct { value: 1, s: TheStruct { value: 2 } };

    unsafe {
        let _y = &X;
        //[e2024]~^ ERROR creating a shared reference to a mutable static [E0796]
        //[e2021]~^^ ERROR shared reference to mutable static is discouraged [static_mut_refs]

        let _y = &mut X;
        //[e2024]~^ ERROR creating a mutable reference to a mutable static [E0796]
        //[e2021]~^^ ERROR mutable reference to mutable static is discouraged [static_mut_refs]

        let _z = addr_of_mut!(X);

        let _p = addr_of!(X);

        let ref _a = X;
        //[e2024]~^ ERROR creating a shared reference to a mutable static [E0796]
        //[e2021]~^^ ERROR shared reference to mutable static is discouraged [static_mut_refs]

        let (_b, _c) = (&X, &Y);
        //[e2024]~^ ERROR creating a shared reference to a mutable static [E0796]
        //[e2021]~^^ ERROR shared reference to mutable static is discouraged [static_mut_refs]
        //[e2024]~^^^ ERROR creating a shared reference to a mutable static [E0796]
        //[e2021]~^^^^ ERROR shared reference to mutable static is discouraged [static_mut_refs]

        foo(&X);
        //[e2024]~^ ERROR creating a shared reference to a mutable static [E0796]
        //[e2021]~^^ ERROR shared reference to mutable static is discouraged [static_mut_refs]

        static mut Z: &[i32; 3] = &[0, 1, 2];

        let _ = Z.len();
        //[e2024]~^ ERROR creating a shared reference to a mutable static [E0796]
        //[e2021]~^^ ERROR creating a shared reference to mutable static is discouraged [static_mut_refs]
        let _ = Z[0];
        //[e2024]~^ ERROR creating a reference to a mutable static [E0796]
        //[e2021]~^^ ERROR creating a reference to mutable static is discouraged [static_mut_refs]
        let _ = format!("{:?}", Z);
        //[e2024]~^ ERROR creating a shared reference to a mutable static [E0796]
        //[e2021]~^^ ERROR creating a shared reference to mutable static is discouraged [static_mut_refs]

        let _v = &A.value;
        //[e2024]~^ ERROR creating a shared reference to a mutable static [E0796]
        //[e2021]~^^ ERROR creating a shared reference to mutable static is discouraged [static_mut_refs]

        let _s = &A.s.value;
        //[e2024]~^ ERROR creating a shared reference to a mutable static [E0796]
        //[e2021]~^^ ERROR creating a shared reference to mutable static is discouraged [static_mut_refs]

        let ref _v = A.value;
        //[e2024]~^ ERROR creating a shared reference to a mutable static [E0796]
        //[e2021]~^^ ERROR creating a shared reference to mutable static is discouraged [static_mut_refs]

        let _x = bar!(FOO);
    }
}

fn foo<'a>(_x: &'a i32) {}
