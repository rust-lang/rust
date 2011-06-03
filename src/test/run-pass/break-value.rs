// xfail-stage0
fn int_id(int x) -> int {
  ret x;
}

fn main() {
  while(true) {
      int_id(break);
  }
}