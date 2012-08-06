//error-pattern: lower range bound
//error-pattern: non-numeric
//error-pattern: mismatched types

fn main() {
    match 5u {
      6u to 1u => { }
      _ => { }
    };

    match "wow" {
      "bar" to "foo" => { }
    };

    match 5u {
      'c' to 100u => { }
      _ => { }
    };
}
