// xfail-stage0
fn main() {
  auto x = ();
  alt (x) {
    case (()) {
    }
  }
}
