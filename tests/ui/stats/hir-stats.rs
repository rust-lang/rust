// check-pass
// compile-flags: -Zhir-stats
// only-x86_64

// Type layouts sometimes change. When that happens, until the next bootstrap
// bump occurs, stage1 and stage2 will give different outputs for this test.
// Add an `ignore-stage1` comment marker to work around that problem during
// that time.

// The aim here is to include at least one of every different type of top-level
// AST/HIR node reported by `-Zhir-stats`.

#![allow(dead_code)]

use std::arch::asm;
use std::fmt::Debug;
use std::ffi::c_void;

extern "C" { fn f(p: *mut c_void); }

/// An enum.
enum E<'a, T: Copy> { A { t: T }, B(&'a u32) }

trait Go {
    type G: Debug;
    fn go(self) -> u32;
}

impl<'a, T: Copy> Go for E<'a, T> {
    type G = bool;
    fn go(self) -> u32 {
        99
    }
}

fn f2<T>(t: T) where T: Debug {}

fn main() {
    let x = E::A { t: 3 };
    match x {
        E::A { .. } => {}
        _ => {}
    }

    unsafe { asm!("mov rdi, 1"); }
}
