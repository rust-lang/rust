// error-pattern:3:13:3:14
fn main() -> () {
  auto foo = [];
}
// this checks that span_err gets used
