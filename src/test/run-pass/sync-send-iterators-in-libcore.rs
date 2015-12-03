// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// pretty-expanded FIXME #23616

#![allow(warnings)]
#![feature(iter_empty)]
#![feature(iter_once)]
#![feature(iter_unfold)]
#![feature(step_by)]
#![feature(str_escape)]

use std::iter::{empty, once, repeat};

fn is_sync<T>(_: T) where T: Sync {}
fn is_send<T>(_: T) where T: Send {}

macro_rules! all_sync_send {
    ($ctor:expr, $iter:ident) => ({
        let mut x = $ctor;
        is_sync(x.$iter());
        let mut y = $ctor;
        is_send(y.$iter());
    });
    ($ctor:expr, $iter:ident($($param:expr),+)) => ({
        let mut x = $ctor;
        is_sync(x.$iter($( $param ),+));
        let mut y = $ctor;
        is_send(y.$iter($( $param ),+));
    });
    ($ctor:expr, $iter:ident, $($rest:tt)*) => ({
        all_sync_send!($ctor, $iter);
        all_sync_send!($ctor, $($rest)*);
    });
    ($ctor:expr, $iter:ident($($param:expr),+), $($rest:tt)*) => ({
        all_sync_send!($ctor, $iter($( $param ),+));
        all_sync_send!($ctor, $($rest)*);
    });
}

macro_rules! all_sync_send_mutable_ref {
    ($ctor:expr, $($iter:ident),+) => ({
        $(
            let mut x = $ctor;
            is_sync((&mut x).$iter());
            let mut y = $ctor;
            is_send((&mut y).$iter());
        )+
    })
}

macro_rules! is_sync_send {
    ($ctor:expr) => ({
        let x = $ctor;
        is_sync(x);
        let y = $ctor;
        is_send(y);
    })
}

fn main() {
    // for char.rs
    all_sync_send!("Я", escape_default, escape_unicode);

    // for iter.rs
    all_sync_send_mutable_ref!([1], iter);

    // Bytes implements DoubleEndedIterator
    all_sync_send!("a".bytes(), rev);

    let a = [1];
    let b = [2];
    all_sync_send!(a.iter(),
                   cloned,
                   cycle,
                   chain([2].iter()),
                   zip([2].iter()),
                   map(|_| 1),
                   filter(|_| true),
                   filter_map(|_| Some(1)),
                   enumerate,
                   peekable,
                   skip_while(|_| true),
                   take_while(|_| true),
                   skip(1),
                   take(1),
                   scan(1, |_, _| Some(1)),
                   flat_map(|_| b.iter()),
                   fuse,
                   inspect(|_| ()));

    is_sync_send!((1..).step_by(2));
    is_sync_send!((1..2).step_by(2));
    is_sync_send!((1..2));
    is_sync_send!((1..));
    is_sync_send!(repeat(1));
    is_sync_send!(empty::<usize>());
    is_sync_send!(once(1));

    // for option.rs
    // FIXME

    // for result.rs
    // FIXME

    // for slice.rs
    // FIXME

    // for str/mod.rs
    // FIXME
}
