#![allow(
    dangerous_implicit_autorefs,
    clippy::explicit_auto_deref,
    clippy::return_self_not_must_use,
    clippy::useless_vec
)]
#![warn(clippy::deref_addrof)]

fn get_number() -> usize {
    10
}

fn get_reference(n: &usize) -> &usize {
    n
}

#[allow(clippy::double_parens)]
#[allow(unused_variables, unused_parens)]
fn main() {
    let a = 10;
    let aref = &a;

    let b = *&a;
    //~^ deref_addrof

    let b = *&get_number();
    //~^ deref_addrof

    let b = *get_reference(&a);

    let bytes: Vec<usize> = vec![1, 2, 3, 4];
    let b = *&bytes[1..2][0];
    //~^ deref_addrof

    //This produces a suggestion of 'let b = (a);' which
    //will trigger the 'unused_parens' lint
    let b = *&(a);
    //~^ deref_addrof

    let b = *(&a);
    //~^ deref_addrof

    #[rustfmt::skip]
    let b = *((&a));
    //~^ deref_addrof

    let b = *&&a;
    //~^ deref_addrof

    let b = **&aref;
    //~^ deref_addrof

    let _ = unsafe { *core::ptr::addr_of!(a) };

    let _repeat = *&[0; 64];
    //~^ deref_addrof
    // do NOT lint for array as semantic differences with/out `*&`.
    let _arr = *&[0, 1, 2, 3, 4];

    // Do not lint when text comes from macro
    macro_rules! mac {
        (dr) => {
            *&0
        };
        (dr $e:expr) => {
            *&$e
        };
        (r $e:expr) => {
            &$e
        };
    }
    let b = mac!(dr);
    let b = mac!(dr a);
    let b = *mac!(r a);
}

fn issue14386() {
    use std::mem::ManuallyDrop;

    #[derive(Copy, Clone)]
    struct Data {
        num: u64,
    }

    #[derive(Clone, Copy)]
    struct M {
        md: ManuallyDrop<[u8; 4]>,
    }

    union DataWithPadding<'lt> {
        data: ManuallyDrop<Data>,
        prim: ManuallyDrop<u64>,
        padding: [u8; size_of::<Data>()],
        tup: (ManuallyDrop<Data>, ()),
        indirect: M,
        indirect_arr: [M; 2],
        indirect_ref: &'lt mut M,
    }

    let mut a = DataWithPadding {
        padding: [0; size_of::<DataWithPadding>()],
    };
    unsafe {
        (*&mut a.padding) = [1; size_of::<DataWithPadding>()];
        //~^ deref_addrof
        (*&mut a.tup).1 = ();
        //~^ deref_addrof
        **&mut a.prim = 0;
        //~^ deref_addrof

        (*&mut a.data).num = 42;
        //~^ deref_addrof
        (*&mut a.indirect.md)[3] = 1;
        //~^ deref_addrof
        (*&mut a.indirect_arr[1].md)[3] = 1;
        //~^ deref_addrof
        (*&mut a.indirect_ref.md)[3] = 1;
        //~^ deref_addrof

        // Check that raw pointers are properly considered as well
        **&raw mut a.prim = 0;
        //~^ deref_addrof
        (*&raw mut a.data).num = 42;
        //~^ deref_addrof

        // Do not lint, as the dereference happens later, we cannot
        // just remove `&mut`
        (*&mut a.tup).0.num = 42;
    }
}
