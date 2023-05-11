// run-pass
#![allow(illegal_floating_point_literal_pattern)] // FIXME #41620
#![feature(exclusive_range_pattern)]

pub fn main() {
    match 5_usize {
      1_usize..=5_usize => {}
      _ => panic!("should match range"),
    }
    match 1_usize {
        1_usize..5_usize => {}
        _ => panic!("should match range start"),
    }
    match 5_usize {
      6_usize..=7_usize => panic!("shouldn't match range"),
      _ => {}
    }
    match 7_usize {
        6_usize..7_usize => panic!("shouldn't match range end"),
        _ => {},
    }
    match 5_usize {
      1_usize => panic!("should match non-first range"),
      2_usize..=6_usize => {}
      _ => panic!("math is broken")
    }
    match 'c' {
      'a'..='z' => {}
      _ => panic!("should support char ranges")
    }
    match -3 {
      -7..=5 => {}
      _ => panic!("should match signed range")
    }
    match 3.0f64 {
      1.0..=5.0 => {}
      _ => panic!("should match float range")
    }
    match -1.5f64 {
      -3.6..=3.6 => {}
      _ => panic!("should match negative float range")
    }
    match 3.5 {
        0.0..3.5 => panic!("should not match the range end"),
        _ => {},
    }
    match 0.0 {
        0.0..3.5 => {},
        _ => panic!("should match the range start"),
    }
}
