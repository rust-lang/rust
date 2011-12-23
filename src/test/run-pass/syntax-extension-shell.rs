// xfail-test
fn main() {
  auto s = #shell { uname -a };
  log(debug, s);
}
