//xfail-stage0
//xfail-stage1
//xfail-stage2
//xfail-stage3
fn int_id(int x) -> int {
  ret x;
}

fn main() {
  while(true) {
      int_id(break);
  }
}