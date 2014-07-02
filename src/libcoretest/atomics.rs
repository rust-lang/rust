// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use core::atomics::*;

#[test]
fn bool_() {
    let a = AtomicBool::new(false);
    assert_eq!(a.compare_and_swap(false, true, SeqCst), false);
    assert_eq!(a.compare_and_swap(false, true, SeqCst), true);

    a.store(false, SeqCst);
    assert_eq!(a.compare_and_swap(false, true, SeqCst), false);
}

#[test]
fn bool_and() {
    let a = AtomicBool::new(true);
    assert_eq!(a.fetch_and(false, SeqCst),true);
    assert_eq!(a.load(SeqCst),false);
}

#[test]
fn uint_and() {
    let x = AtomicUint::new(0xf731);
    assert_eq!(x.fetch_and(0x137f, SeqCst), 0xf731);
    assert_eq!(x.load(SeqCst), 0xf731 & 0x137f);
}

#[test]
fn uint_or() {
    let x = AtomicUint::new(0xf731);
    assert_eq!(x.fetch_or(0x137f, SeqCst), 0xf731);
    assert_eq!(x.load(SeqCst), 0xf731 | 0x137f);
}

#[test]
fn uint_xor() {
    let x = AtomicUint::new(0xf731);
    assert_eq!(x.fetch_xor(0x137f, SeqCst), 0xf731);
    assert_eq!(x.load(SeqCst), 0xf731 ^ 0x137f);
}

#[test]
fn int_and() {
    let x = AtomicInt::new(0xf731);
    assert_eq!(x.fetch_and(0x137f, SeqCst), 0xf731);
    assert_eq!(x.load(SeqCst), 0xf731 & 0x137f);
}

#[test]
fn int_or() {
    let x = AtomicInt::new(0xf731);
    assert_eq!(x.fetch_or(0x137f, SeqCst), 0xf731);
    assert_eq!(x.load(SeqCst), 0xf731 | 0x137f);
}

#[test]
fn int_xor() {
    let x = AtomicInt::new(0xf731);
    assert_eq!(x.fetch_xor(0x137f, SeqCst), 0xf731);
    assert_eq!(x.load(SeqCst), 0xf731 ^ 0x137f);
}

static mut S_BOOL : AtomicBool = INIT_ATOMIC_BOOL;
static mut S_INT  : AtomicInt  = INIT_ATOMIC_INT;
static mut S_UINT : AtomicUint = INIT_ATOMIC_UINT;

#[test]
fn static_init() {
    unsafe {
        assert!(!S_BOOL.load(SeqCst));
        assert!(S_INT.load(SeqCst) == 0);
        assert!(S_UINT.load(SeqCst) == 0);
    }
}
