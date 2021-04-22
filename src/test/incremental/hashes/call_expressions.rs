// This test case tests the incremental compilation hash (ICH) implementation
// for function and method call expressions.

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
#![crate_type="rlib"]

fn callee1(_x: u32, _y: i64) {}
fn callee2(_x: u32, _y: i64) {}


// Change Callee (Function)
#[cfg(any(cfail1,cfail4))]
pub fn change_callee_function() {
    callee1(1, 2)
}

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(cfg="cfail2", except="hir_owner_nodes,optimized_mir,typeck")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(cfg="cfail5", except="hir_owner_nodes,optimized_mir,typeck")]
#[rustc_clean(cfg="cfail6")]
pub fn change_callee_function() {
    callee2(1, 2)
}



// Change Argument (Function)
#[cfg(any(cfail1,cfail4))]
pub fn change_argument_function() {
    callee1(1, 2)
}

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(cfg="cfail2", except="hir_owner_nodes,optimized_mir")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(cfg="cfail5", except="hir_owner_nodes,optimized_mir")]
#[rustc_clean(cfg="cfail6")]
pub fn change_argument_function() {
    callee1(1, 3)
}



// Change Callee Indirectly (Function)
mod change_callee_indirectly_function {
    #[cfg(any(cfail1,cfail4))]
    use super::callee1 as callee;
    #[cfg(not(any(cfail1,cfail4)))]
    use super::callee2 as callee;

    #[rustc_clean(except="hir_owner_nodes,typeck", cfg="cfail2")]
    #[rustc_clean(cfg="cfail3")]
    #[rustc_clean(except="hir_owner_nodes,typeck", cfg="cfail5")]
    #[rustc_clean(cfg="cfail6")]
    pub fn change_callee_indirectly_function() {
        callee(1, 2)
    }
}


struct Struct;
impl Struct {
    fn method1(&self, _x: char, _y: bool) {}
    fn method2(&self, _x: char, _y: bool) {}
}

// Change Callee (Method)
#[cfg(any(cfail1,cfail4))]
pub fn change_callee_method() {
    let s = Struct;
    s.method1('x', true);
}

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(cfg="cfail2", except="hir_owner_nodes,optimized_mir,typeck")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(cfg="cfail5", except="hir_owner_nodes,optimized_mir,typeck")]
#[rustc_clean(cfg="cfail6")]
pub fn change_callee_method() {
    let s = Struct;
    s.method2('x', true);
}



// Change Argument (Method)
#[cfg(any(cfail1,cfail4))]
pub fn change_argument_method() {
    let s = Struct;
    s.method1('x', true);
}

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(cfg="cfail2", except="hir_owner_nodes,optimized_mir")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(cfg="cfail5", except="hir_owner_nodes,optimized_mir")]
#[rustc_clean(cfg="cfail6")]
pub fn change_argument_method() {
    let s = Struct;
    s.method1('y', true);
}



// Change Callee (Method, UFCS)
#[cfg(any(cfail1,cfail4))]
pub fn change_ufcs_callee_method() {
    let s = Struct;
    Struct::method1(&s, 'x', true);
}

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(cfg="cfail2", except="hir_owner_nodes,optimized_mir,typeck")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(cfg="cfail5", except="hir_owner_nodes,optimized_mir,typeck")]
#[rustc_clean(cfg="cfail6")]
pub fn change_ufcs_callee_method() {
    let s = Struct;
    Struct::method2(&s, 'x', true);
}



// Change Argument (Method, UFCS)
#[cfg(any(cfail1,cfail4))]
pub fn change_argument_method_ufcs() {
    let s = Struct;
    Struct::method1(&s, 'x', true);
}

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(cfg="cfail2", except="hir_owner_nodes,optimized_mir")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(cfg="cfail5", except="hir_owner_nodes,optimized_mir")]
#[rustc_clean(cfg="cfail6")]
pub fn change_argument_method_ufcs() {
    let s = Struct;
    Struct::method1(&s, 'x',false);
}



// Change To UFCS
#[cfg(any(cfail1,cfail4))]
pub fn change_to_ufcs() {
    let s = Struct;
    s.method1('x', true); // ------
}

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(cfg="cfail2", except="hir_owner_nodes,optimized_mir,typeck")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(cfg="cfail5", except="hir_owner_nodes,optimized_mir,typeck")]
#[rustc_clean(cfg="cfail6")]
// One might think this would be expanded in the hir_owner_nodes/Mir, but it actually
// results in slightly different hir_owner/Mir.
pub fn change_to_ufcs() {
    let s = Struct;
    Struct::method1(&s, 'x', true);
}


struct Struct2;
impl Struct2 {
    fn method1(&self, _x: char, _y: bool) {}
}

// Change UFCS Callee Indirectly
pub mod change_ufcs_callee_indirectly {
    #[cfg(any(cfail1,cfail4))]
    use super::Struct as Struct;
    #[cfg(not(any(cfail1,cfail4)))]
    use super::Struct2 as Struct;

    #[rustc_clean(cfg="cfail2", except="hir_owner_nodes,optimized_mir,typeck")]
    #[rustc_clean(cfg="cfail3")]
    #[rustc_clean(cfg="cfail5", except="hir_owner_nodes,optimized_mir,typeck")]
    #[rustc_clean(cfg="cfail6")]
    pub fn change_ufcs_callee_indirectly() {
        let s = Struct;
        Struct::method1(&s, 'q', false)
    }
}
