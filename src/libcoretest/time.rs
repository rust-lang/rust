// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use core::time::Duration;

#[test]
fn creation() {
    assert!(Duration::from_secs(1) != Duration::from_secs(0));
    assert_eq!(Duration::from_secs(1) + Duration::from_secs(2),
               Duration::from_secs(3));
    assert_eq!(Duration::from_millis(10) + Duration::from_secs(4),
               Duration::new(4, 10 * 1_000_000));
    assert_eq!(Duration::from_millis(4000), Duration::new(4, 0));
}

#[test]
fn secs() {
    assert_eq!(Duration::new(0, 0).as_secs(), 0);
    assert_eq!(Duration::from_secs(1).as_secs(), 1);
    assert_eq!(Duration::from_millis(999).as_secs(), 0);
    assert_eq!(Duration::from_millis(1001).as_secs(), 1);
}

#[test]
fn nanos() {
    assert_eq!(Duration::new(0, 0).subsec_nanos(), 0);
    assert_eq!(Duration::new(0, 5).subsec_nanos(), 5);
    assert_eq!(Duration::new(0, 1_000_000_001).subsec_nanos(), 1);
    assert_eq!(Duration::from_secs(1).subsec_nanos(), 0);
    assert_eq!(Duration::from_millis(999).subsec_nanos(), 999 * 1_000_000);
    assert_eq!(Duration::from_millis(1001).subsec_nanos(), 1 * 1_000_000);
}

#[test]
fn add() {
    assert_eq!(Duration::new(0, 0) + Duration::new(0, 1),
               Duration::new(0, 1));
    assert_eq!(Duration::new(0, 500_000_000) + Duration::new(0, 500_000_001),
               Duration::new(1, 1));
}

#[test]
fn sub() {
    assert_eq!(Duration::new(0, 1) - Duration::new(0, 0),
               Duration::new(0, 1));
    assert_eq!(Duration::new(0, 500_000_001) - Duration::new(0, 500_000_000),
               Duration::new(0, 1));
    assert_eq!(Duration::new(1, 0) - Duration::new(0, 1),
               Duration::new(0, 999_999_999));
}

#[test] #[should_panic]
fn sub_bad1() {
    Duration::new(0, 0) - Duration::new(0, 1);
}

#[test] #[should_panic]
fn sub_bad2() {
    Duration::new(0, 0) - Duration::new(1, 0);
}

#[test]
fn mul() {
    assert_eq!(Duration::new(0, 1) * 2, Duration::new(0, 2));
    assert_eq!(Duration::new(1, 1) * 3, Duration::new(3, 3));
    assert_eq!(Duration::new(0, 500_000_001) * 4, Duration::new(2, 4));
    assert_eq!(Duration::new(0, 500_000_001) * 4000,
               Duration::new(2000, 4000));
}

#[test]
fn div() {
    assert_eq!(Duration::new(0, 1) / 2, Duration::new(0, 0));
    assert_eq!(Duration::new(1, 1) / 3, Duration::new(0, 333_333_333));
    assert_eq!(Duration::new(99, 999_999_000) / 100,
               Duration::new(0, 999_999_990));
}
