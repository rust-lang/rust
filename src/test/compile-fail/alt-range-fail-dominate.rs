//error-pattern: unreachable
//error-pattern: unreachable
//error-pattern: unreachable
//error-pattern: unreachable
//error-pattern: unreachable

fn main() {
    match 5u {
      1u .. 10u => { }
      5u .. 6u => { }
      _ => {}
    };

    match 5u {
      3u .. 6u => { }
      4u .. 6u => { }
      _ => {}
    };

    match 5u {
      4u .. 6u => { }
      4u .. 6u => { }
      _ => {}
    };

    match 'c' {
      'A' .. 'z' => {}
      'a' .. 'z' => {}
      _ => {}
    };

    match 1.0 {
      0.01 .. 6.5 => {}
      0.02 => {}
      _ => {}
    };
}