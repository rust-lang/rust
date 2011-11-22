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
      3u to 6u { }
      4u to 6u { }
    };

    alt 5u {
      4u to 6u { }
      4u to 6u { }
    };

    alt 'c' {
      'A' to 'z' {}
      'a' to 'z' {}
    };

    alt 1.0 {
      0.01 to 6.5 {}
      0.02 {}
    };
}