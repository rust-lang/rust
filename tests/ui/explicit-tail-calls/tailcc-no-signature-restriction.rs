//@ run-pass
//@ ignore-backends: gcc
//@ min-llvm-version: 22
//@ revisions: x86 x86_64 aarch64
//
//@ [x86] only-x86
//@ [x86_64] only-x86_64
//@ [aarch64] only-aarch64
#![feature(explicit_tail_calls, rust_tail_cc)]

pub extern "tail" fn add() -> u64 {
    #[inline(never)]
    extern "tail" fn add(a: u64, b: u64) -> u64 {
        a.wrapping_add(b)
    }

    become add(1, 2);
}

pub extern "tail" fn pass_struct(a: u64, d: u64) -> u64 {
    #[derive(Clone, Copy)]
    pub struct Large {
        pub a: u64,
        pub b: u64,
        pub c: u64,
        pub d: u64,
    }

    #[inline(never)]
    extern "tail" fn add(large: Large) -> u64 {
        let _ = large.b;
        let _ = large.c;
        large.a.wrapping_add(large.d)
    }

    let large = Large { a, b: 0xBBBB_BBBB_BBBB_BBBB, c: 0xCCCC_CCCC_CCCC_CCCC, d };
    become add(large);
}

fn main() {
    assert_eq!(add(), 3);

    assert_eq!(pass_struct(5, 6), 5 + 6);
}
