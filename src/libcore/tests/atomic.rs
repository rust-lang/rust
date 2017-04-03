// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use core::sync::atomic::*;
use core::sync::atomic::Ordering::SeqCst;

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
    let x = AtomicUsize::new(0xf731);
    assert_eq!(x.fetch_and(0x137f, SeqCst), 0xf731);
    assert_eq!(x.load(SeqCst), 0xf731 & 0x137f);
}

#[test]
fn uint_or() {
    let x = AtomicUsize::new(0xf731);
    assert_eq!(x.fetch_or(0x137f, SeqCst), 0xf731);
    assert_eq!(x.load(SeqCst), 0xf731 | 0x137f);
}

#[test]
fn uint_xor() {
    let x = AtomicUsize::new(0xf731);
    assert_eq!(x.fetch_xor(0x137f, SeqCst), 0xf731);
    assert_eq!(x.load(SeqCst), 0xf731 ^ 0x137f);
}

#[test]
fn int_and() {
    let x = AtomicIsize::new(0xf731);
    assert_eq!(x.fetch_and(0x137f, SeqCst), 0xf731);
    assert_eq!(x.load(SeqCst), 0xf731 & 0x137f);
}

#[test]
fn int_or() {
    let x = AtomicIsize::new(0xf731);
    assert_eq!(x.fetch_or(0x137f, SeqCst), 0xf731);
    assert_eq!(x.load(SeqCst), 0xf731 | 0x137f);
}

#[test]
fn int_xor() {
    let x = AtomicIsize::new(0xf731);
    assert_eq!(x.fetch_xor(0x137f, SeqCst), 0xf731);
    assert_eq!(x.load(SeqCst), 0xf731 ^ 0x137f);
}

static S_FALSE: AtomicBool = AtomicBool::new(false);
static S_TRUE: AtomicBool = AtomicBool::new(true);
static S_INT: AtomicIsize  = AtomicIsize::new(0);
static S_UINT: AtomicUsize = AtomicUsize::new(0);

#[test]
fn static_init() {
    assert!(!S_FALSE.load(SeqCst));
    assert!(S_TRUE.load(SeqCst));
    assert!(S_INT.load(SeqCst) == 0);
    assert!(S_UINT.load(SeqCst) == 0);
}
