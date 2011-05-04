// xfail-boot
// xfail-boot
// xfail-stage0
// xfail-stage1
// xfail-stage2
fn main() {
  let vec[int] v = vec(1,2,3,4,5);
  auto v2 = v.(1,2);
  assert (v2.(0) == 2);
  assert (v2.(1) == 3);
}