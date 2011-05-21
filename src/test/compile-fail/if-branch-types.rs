// error-pattern:mismatched types

fn main() {
  auto x = if (true) {
    10
  } else {
    10u
  };
}
