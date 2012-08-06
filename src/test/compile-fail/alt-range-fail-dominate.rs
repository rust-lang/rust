//error-pattern: unreachable
//error-pattern: unreachable
//error-pattern: unreachable
//error-pattern: unreachable
//error-pattern: unreachable

fn main() {
    match check 5u {
      1u to 10u => { }
      5u to 6u => { }
    };

    match check 5u {
      3u to 6u => { }
      4u to 6u => { }
    };

    match check 5u {
      4u to 6u => { }
      4u to 6u => { }
    };

    match check 'c' {
      'A' to 'z' => {}
      'a' to 'z' => {}
    };

    match check 1.0 {
      0.01 to 6.5 => {}
      0.02 => {}
    };
}