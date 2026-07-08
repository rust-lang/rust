//@ check-pass

#![feature(associated_const_underscore)]

use std::mem;

struct Struct<T>(T);

impl<T> Struct<T> {
    const _: () = {
        let _: Option<Self>;
    };

    const _: i16 = 0;
}

impl Struct<i16> {
    const _: () = ();
}

type Field<'a, T> = &'a mut T;

struct Thing<'a, T> {
    field: Field<'a, T>,
}

impl<'a, T: Eq> Thing<'a, T> {
    const _: () = {
        fn require_outlives<'a, T: 'a>() {}
        let _ = require_outlives::<'a, T>;
        let _: Option<Self>;
    };
}

struct Unit;

impl Unit {
    // An associated const named `_` is not evaluated unless it is accessed.
    // However, associated consts named `_` cannot be accessed, so this should pass.
    const _: () = panic!();
    const _: () = assert!(false);
    const _: [(); {
        {
            assert!(std::mem::size_of::<Self>() == 0)
        };
        0
    }] = [];
    const _: [(); {
        {
            assert!(true)
        };
        0
    }] = [];
}

struct Generic<T>(T);

impl<T> Generic<T> {
    const _: () = assert!(mem::size_of::<T>() % 2 == 0);
}

fn main() {
    let _ = Unit;
    let _ = Generic([0u8; 3]);
}

trait CfgTrait {
    #[cfg(any())]
    const _: () = ();
}

impl CfgTrait for Unit {
    #[cfg(any())]
    const _: () = ();
}
