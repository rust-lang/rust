// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// This test illustrates use of `box (<place>) <value>` syntax to
// allocate and initialize user-defined smart-pointer types.
//
// (Note that the code itself is somewhat naive, especially since you
//  wouldn't want to box f32*4 in this way in practice.)

use std::cell::Cell;
use std::kinds::marker;
use std::mem;
use std::ops::{Placer,PlacementAgent};

pub fn main() {
    let ref mut arena = PoolFloat4::new();
    inner(arena);

    let expecting = [true, true, true, true, true];
    assert_eq!(arena.avail_snapshot().as_slice(), expecting.as_slice());
}

fn inner(arena: &mut PoolFloat4) {
    fn avail<'a>(arena: &PoolFloat4) -> [bool, ..5] { arena.avail_snapshot() }

    let expecting = [true, true, true, true, true];
    assert_eq!(avail(arena).as_slice(), expecting.as_slice());

    let a: BoxFloat4 = box (*arena) [1.0, 2.0, 3.0, 4.0];

    let expecting = [false, true, true, true, true];
    assert_eq!(avail(arena).as_slice(), expecting.as_slice());
    assert_eq!(a.xyzw(), (1.0, 2.0, 3.0, 4.0));

    let mut order = vec![];

    let b: BoxFloat4 = box (*{ order.push("arena"); &mut *arena }) [
        { order.push("10.0"); 10.0 },
        { order.push("20.0"); 20.0 },
        { order.push("30.0"); 30.0 },
        { order.push("40.0"); 40.0 },
        ];

    let expecting = [false, false, true, true, true];
    assert_eq!(avail(arena).as_slice(), expecting.as_slice());
    assert_eq!(a.xyzw(), (1.0, 2.0, 3.0, 4.0));
    assert_eq!(b.xyzw(), (10.0, 20.0, 30.0, 40.0));
    assert_eq!(order, vec!["arena", "10.0", "20.0", "30.0", "40.0"]);

    {
        let c: BoxFloat4 = box (*arena) [100.0, 200.0, 300.0, 400.0];
        let expecting = [false, false, false, true, true];
        assert_eq!(avail(arena).as_slice(), expecting.as_slice());
        assert_eq!(a.xyzw(), (1.0, 2.0, 3.0, 4.0));
        assert_eq!(b.xyzw(), (10.0, 20.0, 30.0, 40.0));
        assert_eq!(c.xyzw(), (100.0, 200.0, 300.0, 400.0));
    }

    let expecting = [false, false, true, true, true];
    assert_eq!(avail(arena).as_slice(), expecting.as_slice());
    assert_eq!(a.xyzw(), (1.0, 2.0, 3.0, 4.0));
    assert_eq!(b.xyzw(), (10.0, 20.0, 30.0, 40.0));
}

struct BoxFloat4 {
    arena: *mut PoolFloat4,
    f4: *mut f32,
}

impl BoxFloat4 {
    fn x(&self) -> f32 { unsafe { *self.f4.offset(0) } }
    fn y(&self) -> f32 { unsafe { *self.f4.offset(1) } }
    fn z(&self) -> f32 { unsafe { *self.f4.offset(2) } }
    fn w(&self) -> f32 { unsafe { *self.f4.offset(3) } }

    fn xyzw(&self) -> (f32,f32,f32,f32) {
        (self.x(), self.y(), self.z(), self.w())
    }
}

struct InterimBoxFloat4 {
    arena: *mut PoolFloat4,
    f4: *mut f32,
}

struct PoolFloat4 {
    pool: [f32, ..20],
    avail: [bool, ..5],
    no_copy: marker::NoCopy,
}

impl PoolFloat4 {
    fn new() -> PoolFloat4 {
        let ret = PoolFloat4 {
            pool: [0.0, ..20],
            avail: [true, ..5],
            no_copy: marker::NoCopy,
        };

        ret
    }

    fn avail_snapshot(&self) -> [bool, ..5] {
        self.avail
    }

    fn first_avail(&self) -> Option<uint> {
        for i in range(0u, 5) {
            if self.avail[i] {
                return Some(i);
            }
        }
        None
    }

    fn find_entry(&self, p: *mut f32) -> Option<uint> {
        for i in range(0u, 5) {
            let ptr : &f32 = &self.pool[i*4];
            let cand: *mut f32 = unsafe { mem::transmute(ptr) };
            if p == cand {
                return Some(i);
            }
        }
        None
    }

    unsafe fn mark_freed(&mut self, p: *mut f32) {
        let i = match self.find_entry(p) {
            Some(i) => i,
            None => {
                // (avoiding fail! in drop)
                println!("interim internal consistency failure.");
                return;
            }
        };
        self.avail[i] = true;
        assert_eq!(self.avail[i], true);
    }
}

impl<'a> Placer<[f32, ..4], BoxFloat4, InterimBoxFloat4>
    for PoolFloat4 {
    fn make_place(&mut self) -> InterimBoxFloat4 {
        let i = self.first_avail()
            .unwrap_or_else(|| fail!("exhausted all spots"));

        self.avail[i] = false;
        assert_eq!(self.avail[i], false);
        unsafe {
            let f4 : *mut f32 = mem::transmute(&self.pool[i*4]);
            InterimBoxFloat4 { arena: mem::transmute(self), f4: f4 }
        }
    }
}

impl PlacementAgent<[f32, ..4], BoxFloat4> for InterimBoxFloat4 {
    unsafe fn pointer(&self) -> *mut [f32, ..4] {
        self.f4 as *mut [f32, ..4]
    }

    unsafe fn finalize(self) -> BoxFloat4 {
        let ret = BoxFloat4 { arena: self.arena, f4: self.f4 };
        mem::forget(self);
        ret
    }
}

impl Drop for InterimBoxFloat4 {
    fn drop(&mut self) {
        unsafe {
            let a : &mut PoolFloat4 = mem::transmute(self.arena);
            a.mark_freed(self.f4);
        }
    }
}

impl Drop for BoxFloat4 {
    fn drop(&mut self) {
        unsafe {
            let a : &mut PoolFloat4 = mem::transmute(self.arena);
            a.mark_freed(self.f4);
        }
    }
}
