// xfail-stage0
// Issue #521

fn f() {
  auto x = alt (true) {
    case (true) { 10 }
    case (false) { ret }
  };
}

fn main() { }
