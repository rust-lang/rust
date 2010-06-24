fn main() {
  let vec[int] v = vec(1,2,3,4,5);
  auto v2 = v.(1,2);
  check (v2.(0) == 2);
  check (v2.(1) == 3);
}