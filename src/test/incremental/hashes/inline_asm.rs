// This test case tests the incremental compilation hash (ICH) implementation
// for inline asm.

// The general pattern followed here is: Change one thing between rev1 and rev2
// and make sure that the hash has changed, then change nothing between rev2 and
// rev3 and make sure that the hash has not changed.

// build-pass (FIXME(62277): could be check-pass?)
// revisions: cfail1 cfail2 cfail3 cfail4 cfail5 cfail6
// compile-flags: -Z query-dep-graph
// needs-asm-support
// [cfail1]compile-flags: -Zincremental-ignore-spans
// [cfail2]compile-flags: -Zincremental-ignore-spans
// [cfail3]compile-flags: -Zincremental-ignore-spans
// [cfail4]compile-flags: -Zincremental-relative-spans
// [cfail5]compile-flags: -Zincremental-relative-spans
// [cfail6]compile-flags: -Zincremental-relative-spans

#![allow(warnings)]
#![feature(rustc_attrs)]
#![crate_type="rlib"]

use std::arch::asm;

// Change template
#[cfg(any(cfail1,cfail4))]
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub fn change_template(_a: i32) -> i32 {
    let c: i32;
    unsafe {
        asm!("mov {0}, 1",
             out(reg) c
             );
    }
    c
}

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(cfg="cfail2", except="hir_owner_nodes, optimized_mir")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(cfg="cfail5", except="hir_owner_nodes, optimized_mir")]
#[rustc_clean(cfg="cfail6")]
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub fn change_template(_a: i32) -> i32 {
    let c: i32;
    unsafe {
        asm!("mov {0}, 2",
             out(reg) c
             );
    }
    c
}



// Change output
#[cfg(any(cfail1,cfail4))]
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub fn change_output(a: i32) -> i32 {
    let mut _out1: i32 = 0;
    let mut _out2: i32 = 0;
    unsafe {
        asm!("mov {0}, {1}",
             out(reg) _out1,
             in(reg) a
             );
    }
    _out1
}

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(cfg="cfail2", except="hir_owner_nodes, optimized_mir")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(cfg="cfail5", except="hir_owner_nodes, optimized_mir")]
#[rustc_clean(cfg="cfail6")]
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub fn change_output(a: i32) -> i32 {
    let mut _out1: i32 = 0;
    let mut _out2: i32 = 0;
    unsafe {
        asm!("mov {0}, {1}",
             out(reg) _out2,
             in(reg) a
             );
    }
    _out1
}



// Change input
#[cfg(any(cfail1,cfail4))]
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub fn change_input(_a: i32, _b: i32) -> i32 {
    let _out;
    unsafe {
        asm!("mov {0}, {1}",
             out(reg) _out,
             in(reg) _a
             );
    }
    _out
}

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(cfg="cfail2", except="hir_owner_nodes, optimized_mir")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(cfg="cfail5", except="hir_owner_nodes, optimized_mir")]
#[rustc_clean(cfg="cfail6")]
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub fn change_input(_a: i32, _b: i32) -> i32 {
    let _out;
    unsafe {
        asm!("mov {0}, {1}",
             out(reg) _out,
             in(reg) _b
             );
    }
    _out
}



// Change input constraint
#[cfg(any(cfail1,cfail4))]
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub fn change_input_constraint(_a: i32, _b: i32) -> i32 {
    let _out;
    unsafe {
        asm!("mov {0}, {1}",
             out(reg) _out,
             in(reg) _a,
             in("eax") _b);
    }
    _out
}

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(cfg="cfail2", except="hir_owner_nodes, optimized_mir")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(cfg="cfail5", except="hir_owner_nodes, optimized_mir")]
#[rustc_clean(cfg="cfail6")]
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub fn change_input_constraint(_a: i32, _b: i32) -> i32 {
    let _out;
    unsafe {
        asm!("mov {0}, {1}",
             out(reg) _out,
             in(reg) _a,
             in("ecx") _b);
    }
    _out
}


// Change clobber
#[cfg(any(cfail1,cfail4))]
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub fn change_clobber(_a: i32) -> i32 {
    let _out;
    unsafe {
        asm!("mov {0}, {1}",
             out(reg) _out,
             in(reg) _a,
             lateout("ecx") _
             );
    }
    _out
}

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(cfg="cfail2", except="hir_owner_nodes, optimized_mir")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(cfg="cfail5", except="hir_owner_nodes, optimized_mir")]
#[rustc_clean(cfg="cfail6")]
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub fn change_clobber(_a: i32) -> i32 {
    let _out;
    unsafe {
        asm!("mov {0}, {1}",
             out(reg) _out,
             in(reg) _a,
             lateout("edx") _
             );
    }
    _out
}



// Change options
#[cfg(any(cfail1,cfail4))]
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub fn change_options(_a: i32) -> i32 {
    let _out;
    unsafe {
        asm!("mov {0}, {1}",
             out(reg) _out,
             in(reg) _a,
             options(readonly),
             );
    }
    _out
}

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(cfg="cfail2", except="hir_owner_nodes, optimized_mir")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(cfg="cfail5", except="hir_owner_nodes, optimized_mir")]
#[rustc_clean(cfg="cfail6")]
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub fn change_options(_a: i32) -> i32 {
    let _out;
    unsafe {
        asm!("mov {0}, {1}",
             out(reg) _out,
             in(reg) _a,
             options(nomem   ),
             );
    }
    _out
}
