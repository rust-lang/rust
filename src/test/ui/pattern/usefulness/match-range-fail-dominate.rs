#![deny(unreachable_patterns, overlapping_patterns)]

fn main() {
    match 5 {
      1 ..= 10 => { }
      5 ..= 6 => { }
      //~^ ERROR unreachable pattern
      _ => {}
    };

    match 5 {
      3 ..= 6 => { }
      4 ..= 6 => { }
      //~^ ERROR unreachable pattern
      _ => {}
    };

    match 5 {
      4 ..= 6 => { }
      4 ..= 6 => { }
      //~^ ERROR unreachable pattern
      _ => {}
    };

    match 'c' {
      'A' ..= 'z' => {}
      'a' ..= 'z' => {}
      //~^ ERROR unreachable pattern
      _ => {}
    };

    match 1.0f64 {
      0.01f64 ..= 6.5f64 => {}
      //~^ WARNING floating-point types cannot be used in patterns
      //~| WARNING floating-point types cannot be used in patterns
      //~| WARNING floating-point types cannot be used in patterns
      //~| WARNING floating-point types cannot be used in patterns
      //~| WARNING this was previously accepted by the compiler
      //~| WARNING this was previously accepted by the compiler
      //~| WARNING this was previously accepted by the compiler
      //~| WARNING this was previously accepted by the compiler
      0.02f64 => {} //~ ERROR unreachable pattern
      //~^ WARNING floating-point types cannot be used in patterns
      //~| WARNING floating-point types cannot be used in patterns
      //~| WARNING this was previously accepted by the compiler
      //~| WARNING this was previously accepted by the compiler
      _ => {}
    };
}
