// xfail-stage0
fn main() {
  auto s = #shell { uname -a };
  log s;
}
