//error-pattern: lower range bound
//error-pattern: non-numeric
//error-pattern: mismatched types

fn main() {
    match 5u {
      6u .. 1u => { }
      _ => { }
    };

    match "wow" {
      "bar" .. "foo" => { }
    };

    match 5u {
      'c' .. 100u => { }
      _ => { }
    };
}
