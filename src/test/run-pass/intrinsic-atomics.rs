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
    #[abi = "rust-intrinsic"]
    pub extern {
        pub fn atomic_cxchg(dst: &mut int, old: int, src: int) -> int;
        pub fn atomic_cxchg_acq(dst: &mut int, old: int, src: int) -> int;
        pub fn atomic_cxchg_rel(dst: &mut int, old: int, src: int) -> int;

        pub fn atomic_xchg(dst: &mut int, src: int) -> int;
        pub fn atomic_xchg_acq(dst: &mut int, src: int) -> int;
        pub fn atomic_xchg_rel(dst: &mut int, src: int) -> int;
        
        pub fn atomic_xadd(dst: &mut int, src: int) -> int;
        pub fn atomic_xadd_acq(dst: &mut int, src: int) -> int;
        pub fn atomic_xadd_rel(dst: &mut int, src: int) -> int;
        
        pub fn atomic_xsub(dst: &mut int, src: int) -> int;
        pub fn atomic_xsub_acq(dst: &mut int, src: int) -> int;
        pub fn atomic_xsub_rel(dst: &mut int, src: int) -> int;
    }
}

pub fn main() {
    unsafe {
        let mut x = ~1;

        fail_unless!(rusti::atomic_cxchg(x, 1, 2) == 1);
        fail_unless!(*x == 2);

        fail_unless!(rusti::atomic_cxchg_acq(x, 1, 3) == 2);
        fail_unless!(*x == 2);

        fail_unless!(rusti::atomic_cxchg_rel(x, 2, 1) == 2);
        fail_unless!(*x == 1);

        fail_unless!(rusti::atomic_xchg(x, 0) == 1);
        fail_unless!(*x == 0);

        fail_unless!(rusti::atomic_xchg_acq(x, 1) == 0);
        fail_unless!(*x == 1);

        fail_unless!(rusti::atomic_xchg_rel(x, 0) == 1);
        fail_unless!(*x == 0);

        fail_unless!(rusti::atomic_xadd(x, 1) == 0);
        fail_unless!(rusti::atomic_xadd_acq(x, 1) == 1);
        fail_unless!(rusti::atomic_xadd_rel(x, 1) == 2);
        fail_unless!(*x == 3);

        fail_unless!(rusti::atomic_xsub(x, 1) == 3);
        fail_unless!(rusti::atomic_xsub_acq(x, 1) == 2);
        fail_unless!(rusti::atomic_xsub_rel(x, 1) == 1);
        fail_unless!(*x == 0);
    }
}
