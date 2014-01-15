// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

mod rusti {
    extern "rust-intrinsic" {
        pub fn atomic_cxchg<T>(dst: &mut T, old: T, src: T) -> T;
        pub fn atomic_cxchg_acq<T>(dst: &mut T, old: T, src: T) -> T;
        pub fn atomic_cxchg_rel<T>(dst: &mut T, old: T, src: T) -> T;

        pub fn atomic_load<T>(src: &T) -> T;
        pub fn atomic_load_acq<T>(src: &T) -> T;

        pub fn atomic_store<T>(dst: &mut T, val: T);
        pub fn atomic_store_rel<T>(dst: &mut T, val: T);

        pub fn atomic_xchg<T>(dst: &mut T, src: T) -> T;
        pub fn atomic_xchg_acq<T>(dst: &mut T, src: T) -> T;
        pub fn atomic_xchg_rel<T>(dst: &mut T, src: T) -> T;

        pub fn atomic_xadd<T>(dst: &mut T, src: T) -> T;
        pub fn atomic_xadd_acq<T>(dst: &mut T, src: T) -> T;
        pub fn atomic_xadd_rel<T>(dst: &mut T, src: T) -> T;

        pub fn atomic_xsub<T>(dst: &mut T, src: T) -> T;
        pub fn atomic_xsub_acq<T>(dst: &mut T, src: T) -> T;
        pub fn atomic_xsub_rel<T>(dst: &mut T, src: T) -> T;
    }
}

pub fn main() {
    unsafe {
        let mut x = ~1;

        assert_eq!(rusti::atomic_load(x), 1);
        *x = 5;
        assert_eq!(rusti::atomic_load_acq(x), 5);

        rusti::atomic_store(x,3);
        assert_eq!(*x, 3);
        rusti::atomic_store_rel(x,1);
        assert_eq!(*x, 1);

        assert_eq!(rusti::atomic_cxchg(x, 1, 2), 1);
        assert_eq!(*x, 2);

        assert_eq!(rusti::atomic_cxchg_acq(x, 1, 3), 2);
        assert_eq!(*x, 2);

        assert_eq!(rusti::atomic_cxchg_rel(x, 2, 1), 2);
        assert_eq!(*x, 1);

        assert_eq!(rusti::atomic_xchg(x, 0), 1);
        assert_eq!(*x, 0);

        assert_eq!(rusti::atomic_xchg_acq(x, 1), 0);
        assert_eq!(*x, 1);

        assert_eq!(rusti::atomic_xchg_rel(x, 0), 1);
        assert_eq!(*x, 0);

        assert_eq!(rusti::atomic_xadd(x, 1), 0);
        assert_eq!(rusti::atomic_xadd_acq(x, 1), 1);
        assert_eq!(rusti::atomic_xadd_rel(x, 1), 2);
        assert_eq!(*x, 3);

        assert_eq!(rusti::atomic_xsub(x, 1), 3);
        assert_eq!(rusti::atomic_xsub_acq(x, 1), 2);
        assert_eq!(rusti::atomic_xsub_rel(x, 1), 1);
        assert_eq!(*x, 0);
    }
}
