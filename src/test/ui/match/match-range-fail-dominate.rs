//error-pattern: unreachable
//error-pattern: unreachable
//error-pattern: unreachable
//error-pattern: unreachable
//error-pattern: unreachable

#![deny(unreachable_patterns)]

fn main() {
    match 5 {
      1 ..= 10 => { }
      5 ..= 6 => { }
      _ => {}
    };

    match 5 {
      3 ..= 6 => { }
      4 ..= 6 => { }
      _ => {}
    };

    match 5 {
      4 ..= 6 => { }
      4 ..= 6 => { }
      _ => {}
    };

    match 'c' {
      'A' ..= 'z' => {}
      'a' ..= 'z' => {}
      _ => {}
    };

    match 1.0f64 {
      0.01f64 ..= 6.5f64 => {}
      0.02f64 => {}
      _ => {}
    };
}
