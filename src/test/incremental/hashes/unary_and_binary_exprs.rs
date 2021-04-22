// This test case tests the incremental compilation hash (ICH) implementation
// for unary and binary expressions.

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


// Change constant operand of negation -----------------------------------------
#[cfg(any(cfail1,cfail4))]
pub fn const_negation() -> i32 {
    -10
}

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(except="hir_owner_nodes,optimized_mir", cfg="cfail2")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(except="hir_owner_nodes,optimized_mir", cfg="cfail5")]
#[rustc_clean(cfg="cfail6")]
pub fn const_negation() -> i32 {
     -1
}



// Change constant operand of bitwise not --------------------------------------
#[cfg(any(cfail1,cfail4))]
pub fn const_bitwise_not() -> i32 {
    !100
}

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(except="hir_owner_nodes,optimized_mir", cfg="cfail2")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(except="hir_owner_nodes,optimized_mir", cfg="cfail5")]
#[rustc_clean(cfg="cfail6")]
pub fn const_bitwise_not() -> i32 {
     !99
}



// Change variable operand of negation -----------------------------------------
#[cfg(any(cfail1,cfail4))]
pub fn var_negation(x: i32, y: i32) -> i32 {
    -x
}

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(except="hir_owner_nodes,optimized_mir", cfg="cfail2")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(except="hir_owner_nodes,optimized_mir", cfg="cfail5")]
#[rustc_clean(cfg="cfail6")]
pub fn var_negation(x: i32, y: i32) -> i32 {
    -y
}



// Change variable operand of bitwise not --------------------------------------
#[cfg(any(cfail1,cfail4))]
pub fn var_bitwise_not(x: i32, y: i32) -> i32 {
    !x
}

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(except="hir_owner_nodes,optimized_mir", cfg="cfail2")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(except="hir_owner_nodes,optimized_mir", cfg="cfail5")]
#[rustc_clean(cfg="cfail6")]
pub fn var_bitwise_not(x: i32, y: i32) -> i32 {
    !y
}



// Change variable operand of deref --------------------------------------------
#[cfg(any(cfail1,cfail4))]
pub fn var_deref(x: &i32, y: &i32) -> i32 {
    *x
}

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(except="hir_owner_nodes,optimized_mir", cfg="cfail2")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(except="hir_owner_nodes,optimized_mir", cfg="cfail5")]
#[rustc_clean(cfg="cfail6")]
pub fn var_deref(x: &i32, y: &i32) -> i32 {
    *y
}



// Change first constant operand of addition -----------------------------------
#[cfg(any(cfail1,cfail4))]
pub fn first_const_add() -> i32 {
    1 + 3
}

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(except="hir_owner_nodes,optimized_mir", cfg="cfail2")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(except="hir_owner_nodes,optimized_mir", cfg="cfail5")]
#[rustc_clean(cfg="cfail6")]
pub fn first_const_add() -> i32 {
    2 + 3
}



// Change second constant operand of addition -----------------------------------
#[cfg(any(cfail1,cfail4))]
pub fn second_const_add() -> i32 {
    1 + 2
}

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(except="hir_owner_nodes,optimized_mir", cfg="cfail2")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(except="hir_owner_nodes,optimized_mir", cfg="cfail5")]
#[rustc_clean(cfg="cfail6")]
pub fn second_const_add() -> i32 {
    1 + 3
}



// Change first variable operand of addition -----------------------------------
#[cfg(any(cfail1,cfail4))]
pub fn first_var_add(a: i32, b: i32) -> i32 {
    a + 2
}

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(except="hir_owner_nodes,optimized_mir", cfg="cfail2")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(except="hir_owner_nodes,optimized_mir", cfg="cfail5")]
#[rustc_clean(cfg="cfail6")]
pub fn first_var_add(a: i32, b: i32) -> i32 {
    b + 2
}



// Change second variable operand of addition ----------------------------------
#[cfg(any(cfail1,cfail4))]
pub fn second_var_add(a: i32, b: i32) -> i32 {
    1 + a
}

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(except="hir_owner_nodes,optimized_mir", cfg="cfail2")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(except="hir_owner_nodes,optimized_mir", cfg="cfail5")]
#[rustc_clean(cfg="cfail6")]
pub fn second_var_add(a: i32, b: i32) -> i32 {
    1 + b
}



// Change operator from + to - -------------------------------------------------
#[cfg(any(cfail1,cfail4))]
pub fn plus_to_minus(a: i32) -> i32 {
    1 + a
}

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(except="hir_owner_nodes,optimized_mir", cfg="cfail2")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(except="hir_owner_nodes,optimized_mir", cfg="cfail5")]
#[rustc_clean(cfg="cfail6")]
pub fn plus_to_minus(a: i32) -> i32 {
    1 - a
}



// Change operator from + to * -------------------------------------------------
#[cfg(any(cfail1,cfail4))]
pub fn plus_to_mult(a: i32) -> i32 {
    1 + a
}

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(except="hir_owner_nodes,optimized_mir", cfg="cfail2")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(except="hir_owner_nodes,optimized_mir", cfg="cfail5")]
#[rustc_clean(cfg="cfail6")]
pub fn plus_to_mult(a: i32) -> i32 {
    1 * a
}



// Change operator from + to / -------------------------------------------------
#[cfg(any(cfail1,cfail4))]
pub fn plus_to_div(a: i32) -> i32 {
    1 + a
}

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(except="hir_owner_nodes,optimized_mir", cfg="cfail2")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(except="hir_owner_nodes,optimized_mir", cfg="cfail5")]
#[rustc_clean(cfg="cfail6")]
pub fn plus_to_div(a: i32) -> i32 {
    1 / a
}



// Change operator from + to % -------------------------------------------------
#[cfg(any(cfail1,cfail4))]
pub fn plus_to_mod(a: i32) -> i32 {
    1 + a
}

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(except="hir_owner_nodes,optimized_mir", cfg="cfail2")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(except="hir_owner_nodes,optimized_mir", cfg="cfail5")]
#[rustc_clean(cfg="cfail6")]
pub fn plus_to_mod(a: i32) -> i32 {
    1 % a
}



// Change operator from && to || -----------------------------------------------
#[cfg(any(cfail1,cfail4))]
pub fn and_to_or(a: bool, b: bool) -> bool {
    a && b
}

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(except="hir_owner_nodes,optimized_mir", cfg="cfail2")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(except="hir_owner_nodes,optimized_mir", cfg="cfail5")]
#[rustc_clean(cfg="cfail6")]
pub fn and_to_or(a: bool, b: bool) -> bool {
    a || b
}



// Change operator from & to | -------------------------------------------------
#[cfg(any(cfail1,cfail4))]
pub fn bitwise_and_to_bitwise_or(a: i32) -> i32 {
    1 & a
}

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(except="hir_owner_nodes,optimized_mir", cfg="cfail2")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(except="hir_owner_nodes,optimized_mir", cfg="cfail5")]
#[rustc_clean(cfg="cfail6")]
pub fn bitwise_and_to_bitwise_or(a: i32) -> i32 {
    1 | a
}



// Change operator from & to ^ -------------------------------------------------
#[cfg(any(cfail1,cfail4))]
pub fn bitwise_and_to_bitwise_xor(a: i32) -> i32 {
    1 & a
}

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(except="hir_owner_nodes,optimized_mir", cfg="cfail2")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(except="hir_owner_nodes,optimized_mir", cfg="cfail5")]
#[rustc_clean(cfg="cfail6")]
pub fn bitwise_and_to_bitwise_xor(a: i32) -> i32 {
    1 ^ a
}



// Change operator from & to << ------------------------------------------------
#[cfg(any(cfail1,cfail4))]
pub fn bitwise_and_to_lshift(a: i32) -> i32 {
    a  & 1
}

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(except="hir_owner_nodes,optimized_mir", cfg="cfail2")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(except="hir_owner_nodes,optimized_mir", cfg="cfail5")]
#[rustc_clean(cfg="cfail6")]
pub fn bitwise_and_to_lshift(a: i32) -> i32 {
    a << 1
}



// Change operator from & to >> ------------------------------------------------
#[cfg(any(cfail1,cfail4))]
pub fn bitwise_and_to_rshift(a: i32) -> i32 {
    a  & 1
}

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(except="hir_owner_nodes,optimized_mir", cfg="cfail2")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(except="hir_owner_nodes,optimized_mir", cfg="cfail5")]
#[rustc_clean(cfg="cfail6")]
pub fn bitwise_and_to_rshift(a: i32) -> i32 {
    a >> 1
}



// Change operator from == to != -----------------------------------------------
#[cfg(any(cfail1,cfail4))]
pub fn eq_to_uneq(a: i32) -> bool {
    a == 1
}

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(except="hir_owner_nodes,optimized_mir", cfg="cfail2")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(except="hir_owner_nodes,optimized_mir", cfg="cfail5")]
#[rustc_clean(cfg="cfail6")]
pub fn eq_to_uneq(a: i32) -> bool {
    a != 1
}



// Change operator from == to < ------------------------------------------------
#[cfg(any(cfail1,cfail4))]
pub fn eq_to_lt(a: i32) -> bool {
    a == 1
}

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(except="hir_owner_nodes,optimized_mir", cfg="cfail2")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(except="hir_owner_nodes,optimized_mir", cfg="cfail5")]
#[rustc_clean(cfg="cfail6")]
pub fn eq_to_lt(a: i32) -> bool {
    a  < 1
}



// Change operator from == to > ------------------------------------------------
#[cfg(any(cfail1,cfail4))]
pub fn eq_to_gt(a: i32) -> bool {
    a == 1
}

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(except="hir_owner_nodes,optimized_mir", cfg="cfail2")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(except="hir_owner_nodes,optimized_mir", cfg="cfail5")]
#[rustc_clean(cfg="cfail6")]
pub fn eq_to_gt(a: i32) -> bool {
    a  > 1
}



// Change operator from == to <= -----------------------------------------------
#[cfg(any(cfail1,cfail4))]
pub fn eq_to_le(a: i32) -> bool {
    a == 1
}

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(except="hir_owner_nodes,optimized_mir", cfg="cfail2")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(except="hir_owner_nodes,optimized_mir", cfg="cfail5")]
#[rustc_clean(cfg="cfail6")]
pub fn eq_to_le(a: i32) -> bool {
    a <= 1
}



// Change operator from == to >= -----------------------------------------------
#[cfg(any(cfail1,cfail4))]
pub fn eq_to_ge(a: i32) -> bool {
    a == 1
}

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(except="hir_owner_nodes,optimized_mir", cfg="cfail2")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(except="hir_owner_nodes,optimized_mir", cfg="cfail5")]
#[rustc_clean(cfg="cfail6")]
pub fn eq_to_ge(a: i32) -> bool {
    a >= 1
}



// Change type in cast expression ----------------------------------------------
#[cfg(any(cfail1,cfail4))]
pub fn type_cast(a: u8) -> u64 {
    let b = a as i32;
    let c = b as u64;
    c
}

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(except="hir_owner_nodes,optimized_mir,typeck", cfg="cfail2")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(except="hir_owner_nodes,optimized_mir,typeck", cfg="cfail5")]
#[rustc_clean(cfg="cfail6")]
pub fn type_cast(a: u8) -> u64 {
    let b = a as u32;
    let c = b as u64;
    c
}



// Change value in cast expression ---------------------------------------------
#[cfg(any(cfail1,cfail4))]
pub fn value_cast(a: u32) -> i32 {
    1 as i32
}

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(except="hir_owner_nodes,optimized_mir", cfg="cfail2")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(except="hir_owner_nodes,optimized_mir", cfg="cfail5")]
#[rustc_clean(cfg="cfail6")]
pub fn value_cast(a: u32) -> i32 {
    2 as i32
}



// Change place in assignment --------------------------------------------------
#[cfg(any(cfail1,cfail4))]
pub fn place() -> i32 {
    let mut x = 10;
    let mut y = 11;
    x = 9;
    x
}

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(except="hir_owner_nodes,optimized_mir", cfg="cfail2")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(except="hir_owner_nodes,optimized_mir", cfg="cfail5")]
#[rustc_clean(cfg="cfail6")]
pub fn place() -> i32 {
    let mut x = 10;
    let mut y = 11;
    y = 9;
    x
}



// Change r-value in assignment ------------------------------------------------
#[cfg(any(cfail1,cfail4))]
pub fn rvalue() -> i32 {
    let mut x = 10;
    x = 9;
    x
}

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(except="hir_owner_nodes,optimized_mir", cfg="cfail2")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(except="hir_owner_nodes,optimized_mir", cfg="cfail5")]
#[rustc_clean(cfg="cfail6")]
pub fn rvalue() -> i32 {
    let mut x = 10;
    x = 8;
    x
}



// Change index into slice -----------------------------------------------------
#[cfg(any(cfail1,cfail4))]
pub fn index_to_slice(s: &[u8], i: usize, j: usize) -> u8 {
    s[i]
}

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(except="hir_owner_nodes,optimized_mir", cfg="cfail2")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(except="hir_owner_nodes,optimized_mir", cfg="cfail5")]
#[rustc_clean(cfg="cfail6")]
pub fn index_to_slice(s: &[u8], i: usize, j: usize) -> u8 {
    s[j]
}
