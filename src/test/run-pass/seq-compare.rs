// xfail-boot
fn main() {
  assert ("hello" < "hellr");
  assert ("hello " > "hello");
  assert ("hello" != "there");

  assert (vec(1,2,3,4) > vec(1,2,3));
  assert (vec(1,2,3) < vec(1,2,3,4));
  assert (vec(1,2,4,4) > vec(1,2,3,4));
  assert (vec(1,2,3,4) < vec(1,2,4,4));
  assert (vec(1,2,3) <= vec(1,2,3));
  assert (vec(1,2,3) <= vec(1,2,3,3));
  assert (vec(1,2,3,4) > vec(1,2,3));
  assert (vec(1,2,3) == vec(1,2,3));
  assert (vec(1,2,3) != vec(1,1,3));
}
