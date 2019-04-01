#![allow(illegal_floating_point_literal_pattern)]
#![deny(unreachable_patterns)]

fn main() {
    match 0.0 {
      0.0..=1.0 => {}
      //~^ WARN floating-point types cannot be used in patterns
      //~| WARN this was previously accepted
      //~| WARN hard error
      //~| WARN floating-point types cannot be used in patterns
      //~| WARN this was previously accepted
      //~| WARN hard error
      _ => {} // ok
    }

    match 0.0 { //~ ERROR non-exhaustive patterns
      0.0..=1.0 => {}
      //~^ WARN floating-point types cannot be used in patterns
      //~| WARN this was previously accepted
      //~| WARN hard error
      //~| WARN floating-point types cannot be used in patterns
      //~| WARN this was previously accepted
      //~| WARN hard error
    }
}
