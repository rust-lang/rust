// This test case tests the incremental compilation hash (ICH) implementation
// for inline asm.

// The general pattern followed here is: Change one thing between rev1 and rev2
// and make sure that the hash has changed, then change nothing between rev2 and
// rev3 and make sure that the hash has not changed.

// build-pass (FIXME(62277): could be check-pass?)
// revisions: cfail1 cfail2 cfail3 cfail4 cfail5 cfail6
// compile-flags: -Z query-dep-graph
// [cfail1]compile-flags: -Zincremental-ignore-spans
// [cfail2]compile-flags: -Zincremental-ignore-spans
// [cfail3]compile-flags: -Zincremental-ignore-spans
// [cfail4]compile-flags: -Zincremental-relative-spans
// [cfail5]compile-flags: -Zincremental-relative-spans
// [cfail6]compile-flags: -Zincremental-relative-spans

#![allow(warnings)]
#![feature(rustc_attrs)]
#![feature(llvm_asm)]
#![crate_type="rlib"]



// Change template
#[cfg(any(cfail1,cfail4))]
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub fn change_template(a: i32) -> i32 {
    let c: i32;
    unsafe {
        llvm_asm!("add 1, $0"
                  : "=r"(c)
                  : "0"(a)
                  :
                  :
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
pub fn change_template(a: i32) -> i32 {
    let c: i32;
    unsafe {
        llvm_asm!("add 2, $0"
                  : "=r"(c)
                  : "0"(a)
                  :
                  :
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
        llvm_asm!("add 1, $0"
                  : "=r"(_out1)
                  : "0"(a)
                  :
                  :
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
        llvm_asm!("add 1, $0"
                  : "=r"(_out2)
                  : "0"(a)
                  :
                  :
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
        llvm_asm!("add 1, $0"
                  : "=r"(_out)
                  : "0"(_a)
                  :
                  :
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
        llvm_asm!("add 1, $0"
                  : "=r"(_out)
                  : "0"(_b)
                  :
                  :
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
        llvm_asm!("add 1, $0"
                  : "=r"(_out)
                  : "0"(_a), "r"(_b)
                  :
                  :
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
pub fn change_input_constraint(_a: i32, _b: i32) -> i32 {
    let _out;
    unsafe {
        llvm_asm!("add 1, $0"
                  : "=r"(_out)
                  : "r"(_a), "0"(_b)
                  :
                  :
                  );
    }
    _out
}



// Change clobber
#[cfg(any(cfail1,cfail4))]
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub fn change_clobber(_a: i32) -> i32 {
    let _out;
    unsafe {
        llvm_asm!("add 1, $0"
                  : "=r"(_out)
                  : "0"(_a)
                  :/*--*/
                  :
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
        llvm_asm!("add 1, $0"
                  : "=r"(_out)
                  : "0"(_a)
                  : "eax"
                  :
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
        llvm_asm!("add 1, $0"
                  : "=r"(_out)
                  : "0"(_a)
                  :
                  :/*-------*/
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
        llvm_asm!("add 1, $0"
                  : "=r"(_out)
                  : "0"(_a)
                  :
                  : "volatile"
                  );
    }
    _out
}
