//@ run-pass
#![allow(unused_braces)]
#![allow(dead_code)]

// Tests for standalone blocks as expressions

fn test_basic() { let rs: bool = { true }; assert!(rs); }

struct RS { v1: isize, v2: isize }

fn test_rec() { let rs = { RS {v1: 10, v2: 20} }; assert_eq!(rs.v2, 20); }

fn test_filled_with_stuff() {
    let rs = { let mut a = 0; while a < 10 { a += 1; } a };
    assert_eq!(rs, 10);
}

pub fn main() { test_basic(); test_rec(); test_filled_with_stuff(); }
