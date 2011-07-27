// xfail-stage0
// xfail-stage1
// xfail-stage2
// xfail-stage3
fn main() {
  auto s = #shell { uname -a };
  log s;
}
