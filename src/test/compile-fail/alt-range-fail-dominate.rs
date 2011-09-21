//error-pattern: unreachable
//error-pattern: unreachable
//error-pattern: unreachable
//error-pattern: unreachable
//error-pattern: unreachable
//error-pattern: unreachable

fn main() {
    alt 5u {
      1u to 10u { }
      5u to 6u { }
    };

    alt 5u {
      4u to 6u { }
      3u to 5u { }
    };

    alt 5u {
      4u to 6u { }
      5u to 7u { }
    };

    alt 'c' {
      'A' to 'z' {}
      'a' to 'z' {}
    };

    alt 1.0 {
      -5.0 to 5.0 {}
      0.0 to 6.5 {}
    };

    alt 1.0 {
      0.02 {}
      0.01 to 6.5 {}
    };
}