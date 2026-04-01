// Make sure we don't mess up the formatting of generic consts

#![feature(generic_const_items)]

const GENERIC<N, const M: usize>: i32 = 0;

const WHERECLAUSE: i32 = 0
where
    i32:;

trait Foo {
    const GENERIC<N, const M: usize>: i32;

    const WHERECLAUSE: i32
    where
        i32:;
}

impl Foo for () {
    const GENERIC<N, const M: usize>: i32 = 0;

    const WHERECLAUSE: i32 = 0
    where
        i32:;
}
