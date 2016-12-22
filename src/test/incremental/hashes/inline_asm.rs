// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


// This test case tests the incremental compilation hash (ICH) implementation
// for inline asm.

// The general pattern followed here is: Change one thing between rev1 and rev2
// and make sure that the hash has changed, then change nothing between rev2 and
// rev3 and make sure that the hash has not changed.

// must-compile-successfully
// revisions: cfail1 cfail2 cfail3
// compile-flags: -Z query-dep-graph

#![allow(warnings)]
#![feature(rustc_attrs)]
#![feature(asm)]
#![crate_type="rlib"]



// Change template -------------------------------------------------------------
#[cfg(cfail1)]
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn change_template(a: i32) -> i32 {
    let c: i32;
    unsafe {
        asm!("add 1, $0"
             : "=r"(c)
             : "0"(a)
             :
             :
             );
    }
    c
}

#[cfg(not(cfail1))]
#[rustc_clean(label="Hir", cfg="cfail2")]
#[rustc_clean(label="Hir", cfg="cfail3")]
#[rustc_dirty(label="HirBody", cfg="cfail2")]
#[rustc_clean(label="HirBody", cfg="cfail3")]
#[rustc_metadata_clean(cfg="cfail2")]
#[rustc_metadata_clean(cfg="cfail3")]
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn change_template(a: i32) -> i32 {
    let c: i32;
    unsafe {
        asm!("add 2, $0"
             : "=r"(c)
             : "0"(a)
             :
             :
             );
    }
    c
}



// Change output -------------------------------------------------------------
#[cfg(cfail1)]
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn change_output(a: i32) -> i32 {
    let mut _out1: i32 = 0;
    let mut _out2: i32 = 0;
    unsafe {
        asm!("add 1, $0"
             : "=r"(_out1)
             : "0"(a)
             :
             :
             );
    }
    _out1
}

#[cfg(not(cfail1))]
#[rustc_clean(label="Hir", cfg="cfail2")]
#[rustc_clean(label="Hir", cfg="cfail3")]
#[rustc_dirty(label="HirBody", cfg="cfail2")]
#[rustc_clean(label="HirBody", cfg="cfail3")]
#[rustc_metadata_clean(cfg="cfail2")]
#[rustc_metadata_clean(cfg="cfail3")]
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn change_output(a: i32) -> i32 {
    let mut _out1: i32 = 0;
    let mut _out2: i32 = 0;
    unsafe {
        asm!("add 1, $0"
             : "=r"(_out2)
             : "0"(a)
             :
             :
             );
    }
    _out1
}



// Change input -------------------------------------------------------------
#[cfg(cfail1)]
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn change_input(_a: i32, _b: i32) -> i32 {
    let _out;
    unsafe {
        asm!("add 1, $0"
             : "=r"(_out)
             : "0"(_a)
             :
             :
             );
    }
    _out
}

#[cfg(not(cfail1))]
#[rustc_clean(label="Hir", cfg="cfail2")]
#[rustc_clean(label="Hir", cfg="cfail3")]
#[rustc_dirty(label="HirBody", cfg="cfail2")]
#[rustc_clean(label="HirBody", cfg="cfail3")]
#[rustc_metadata_clean(cfg="cfail2")]
#[rustc_metadata_clean(cfg="cfail3")]
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn change_input(_a: i32, _b: i32) -> i32 {
    let _out;
    unsafe {
        asm!("add 1, $0"
             : "=r"(_out)
             : "0"(_b)
             :
             :
             );
    }
    _out
}



// Change input constraint -----------------------------------------------------
#[cfg(cfail1)]
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn change_input_constraint(_a: i32, _b: i32) -> i32 {
    let _out;
    unsafe {
        asm!("add 1, $0"
             : "=r"(_out)
             : "0"(_a), "r"(_b)
             :
             :
             );
    }
    _out
}

#[cfg(not(cfail1))]
#[rustc_clean(label="Hir", cfg="cfail2")]
#[rustc_clean(label="Hir", cfg="cfail3")]
#[rustc_dirty(label="HirBody", cfg="cfail2")]
#[rustc_clean(label="HirBody", cfg="cfail3")]
#[rustc_metadata_clean(cfg="cfail2")]
#[rustc_metadata_clean(cfg="cfail3")]
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn change_input_constraint(_a: i32, _b: i32) -> i32 {
    let _out;
    unsafe {
        asm!("add 1, $0"
             : "=r"(_out)
             : "r"(_a), "0"(_b)
             :
             :
             );
    }
    _out
}



// Change clobber --------------------------------------------------------------
#[cfg(cfail1)]
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn change_clobber(_a: i32) -> i32 {
    let _out;
    unsafe {
        asm!("add 1, $0"
             : "=r"(_out)
             : "0"(_a)
             :
             :
             );
    }
    _out
}

#[cfg(not(cfail1))]
#[rustc_clean(label="Hir", cfg="cfail2")]
#[rustc_clean(label="Hir", cfg="cfail3")]
#[rustc_dirty(label="HirBody", cfg="cfail2")]
#[rustc_clean(label="HirBody", cfg="cfail3")]
#[rustc_metadata_clean(cfg="cfail2")]
#[rustc_metadata_clean(cfg="cfail3")]
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn change_clobber(_a: i32) -> i32 {
    let _out;
    unsafe {
        asm!("add 1, $0"
             : "=r"(_out)
             : "0"(_a)
             : "eax"
             :
             );
    }
    _out
}



// Change options --------------------------------------------------------------
#[cfg(cfail1)]
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn change_options(_a: i32) -> i32 {
    let _out;
    unsafe {
        asm!("add 1, $0"
             : "=r"(_out)
             : "0"(_a)
             :
             :
             );
    }
    _out
}

#[cfg(not(cfail1))]
#[rustc_clean(label="Hir", cfg="cfail2")]
#[rustc_clean(label="Hir", cfg="cfail3")]
#[rustc_dirty(label="HirBody", cfg="cfail2")]
#[rustc_clean(label="HirBody", cfg="cfail3")]
#[rustc_metadata_clean(cfg="cfail2")]
#[rustc_metadata_clean(cfg="cfail3")]
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn change_options(_a: i32) -> i32 {
    let _out;
    unsafe {
        asm!("add 1, $0"
             : "=r"(_out)
             : "0"(_a)
             :
             : "volatile"
             );
    }
    _out
}



