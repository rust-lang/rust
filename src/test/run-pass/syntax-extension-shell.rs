// xfail-test
fn main() {
  auto s = #shell { uname -a };
  log_full(core::debug, s);
}
