// error-pattern:3:24:3:25
fn main() -> () {
  let @vec[uint] foo = @[];
}
// this checks that span_err gets used
