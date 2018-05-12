// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
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
    assert_eq!(Duration::new(0, 500_000_005).as_secs(), 0);
    assert_eq!(Duration::new(0, 1_050_000_001).as_secs(), 1);
    assert_eq!(Duration::from_secs(1).as_secs(), 1);
    assert_eq!(Duration::from_millis(999).as_secs(), 0);
    assert_eq!(Duration::from_millis(1001).as_secs(), 1);
    assert_eq!(Duration::from_micros(999_999).as_secs(), 0);
    assert_eq!(Duration::from_micros(1_000_001).as_secs(), 1);
    assert_eq!(Duration::from_nanos(999_999_999).as_secs(), 0);
    assert_eq!(Duration::from_nanos(1_000_000_001).as_secs(), 1);
}

#[test]
fn millis() {
    assert_eq!(Duration::new(0, 0).subsec_millis(), 0);
    assert_eq!(Duration::new(0, 500_000_005).subsec_millis(), 500);
    assert_eq!(Duration::new(0, 1_050_000_001).subsec_millis(), 50);
    assert_eq!(Duration::from_secs(1).subsec_millis(), 0);
    assert_eq!(Duration::from_millis(999).subsec_millis(), 999);
    assert_eq!(Duration::from_millis(1001).subsec_millis(), 1);
    assert_eq!(Duration::from_micros(999_999).subsec_millis(), 999);
    assert_eq!(Duration::from_micros(1_001_000).subsec_millis(), 1);
    assert_eq!(Duration::from_nanos(999_999_999).subsec_millis(), 999);
    assert_eq!(Duration::from_nanos(1_001_000_000).subsec_millis(), 1);
}

#[test]
fn micros() {
    assert_eq!(Duration::new(0, 0).subsec_micros(), 0);
    assert_eq!(Duration::new(0, 500_000_005).subsec_micros(), 500_000);
    assert_eq!(Duration::new(0, 1_050_000_001).subsec_micros(), 50_000);
    assert_eq!(Duration::from_secs(1).subsec_micros(), 0);
    assert_eq!(Duration::from_millis(999).subsec_micros(), 999_000);
    assert_eq!(Duration::from_millis(1001).subsec_micros(), 1_000);
    assert_eq!(Duration::from_micros(999_999).subsec_micros(), 999_999);
    assert_eq!(Duration::from_micros(1_000_001).subsec_micros(), 1);
    assert_eq!(Duration::from_nanos(999_999_999).subsec_micros(), 999_999);
    assert_eq!(Duration::from_nanos(1_000_001_000).subsec_micros(), 1);
}

#[test]
fn nanos() {
    assert_eq!(Duration::new(0, 0).subsec_nanos(), 0);
    assert_eq!(Duration::new(0, 5).subsec_nanos(), 5);
    assert_eq!(Duration::new(0, 1_000_000_001).subsec_nanos(), 1);
    assert_eq!(Duration::from_secs(1).subsec_nanos(), 0);
    assert_eq!(Duration::from_millis(999).subsec_nanos(), 999_000_000);
    assert_eq!(Duration::from_millis(1001).subsec_nanos(), 1_000_000);
    assert_eq!(Duration::from_micros(999_999).subsec_nanos(), 999_999_000);
    assert_eq!(Duration::from_micros(1_000_001).subsec_nanos(), 1000);
    assert_eq!(Duration::from_nanos(999_999_999).subsec_nanos(), 999_999_999);
    assert_eq!(Duration::from_nanos(1_000_000_001).subsec_nanos(), 1);
}

#[test]
fn add() {
    assert_eq!(Duration::new(0, 0) + Duration::new(0, 1),
               Duration::new(0, 1));
    assert_eq!(Duration::new(0, 500_000_000) + Duration::new(0, 500_000_001),
               Duration::new(1, 1));
}

#[test]
fn checked_add() {
    assert_eq!(Duration::new(0, 0).checked_add(Duration::new(0, 1)),
               Some(Duration::new(0, 1)));
    assert_eq!(Duration::new(0, 500_000_000).checked_add(Duration::new(0, 500_000_001)),
               Some(Duration::new(1, 1)));
    assert_eq!(Duration::new(1, 0).checked_add(Duration::new(::core::u64::MAX, 0)), None);
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

#[test]
fn checked_sub() {
    let zero = Duration::new(0, 0);
    let one_nano = Duration::new(0, 1);
    let one_sec = Duration::new(1, 0);
    assert_eq!(one_nano.checked_sub(zero), Some(Duration::new(0, 1)));
    assert_eq!(one_sec.checked_sub(one_nano),
               Some(Duration::new(0, 999_999_999)));
    assert_eq!(zero.checked_sub(one_nano), None);
    assert_eq!(zero.checked_sub(one_sec), None);
}

#[test]
#[should_panic]
fn sub_bad1() {
    let _ = Duration::new(0, 0) - Duration::new(0, 1);
}

#[test]
#[should_panic]
fn sub_bad2() {
    let _ = Duration::new(0, 0) - Duration::new(1, 0);
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
fn checked_mul() {
    assert_eq!(Duration::new(0, 1).checked_mul(2), Some(Duration::new(0, 2)));
    assert_eq!(Duration::new(1, 1).checked_mul(3), Some(Duration::new(3, 3)));
    assert_eq!(Duration::new(0, 500_000_001).checked_mul(4), Some(Duration::new(2, 4)));
    assert_eq!(Duration::new(0, 500_000_001).checked_mul(4000),
               Some(Duration::new(2000, 4000)));
    assert_eq!(Duration::new(::core::u64::MAX - 1, 0).checked_mul(2), None);
}

#[test]
fn div() {
    assert_eq!(Duration::new(0, 1) / 2, Duration::new(0, 0));
    assert_eq!(Duration::new(1, 1) / 3, Duration::new(0, 333_333_333));
    assert_eq!(Duration::new(99, 999_999_000) / 100,
               Duration::new(0, 999_999_990));
}

#[test]
fn checked_div() {
    assert_eq!(Duration::new(2, 0).checked_div(2), Some(Duration::new(1, 0)));
    assert_eq!(Duration::new(1, 0).checked_div(2), Some(Duration::new(0, 500_000_000)));
    assert_eq!(Duration::new(2, 0).checked_div(0), None);
}
