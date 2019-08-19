// run-pass
#![allow(illegal_floating_point_literal_pattern)] // FIXME #41620
#![allow(ellipsis_inclusive_range_patterns)]

// regression test for the model lexer handling the DOTDOTDOT syntax (#15877)


pub fn main() {
    match 5_usize {
      1_usize...5_usize => {}
      _ => panic!("should match range"),
    }
    match 5_usize {
      6_usize...7_usize => panic!("shouldn't match range"),
      _ => {}
    }
    match 5_usize {
      1_usize => panic!("should match non-first range"),
      2_usize...6_usize => {}
      _ => panic!("math is broken")
    }
    match 'c' {
      'a'...'z' => {}
      _ => panic!("should support char ranges")
    }
    match -3_isize {
      -7...5 => {}
      _ => panic!("should match signed range")
    }
    match 3.0f64 {
      1.0...5.0 => {}
      _ => panic!("should match float range")
    }
    match -1.5f64 {
      -3.6...3.6 => {}
      _ => panic!("should match negative float range")
    }
}
