fn main() {
  check ("hello" < "hellr");
  check ("hello " > "hello");
  check ("hello" != "there");

  check (vec(1,2,3,4) > vec(1,2,3));
  check (vec(1,2,3) < vec(1,2,3,4));
  check (vec(1,2,4,4) > vec(1,2,3,4));
  check (vec(1,2,3,4) < vec(1,2,4,4));
  check (vec(1,2,3) <= vec(1,2,3));
  check (vec(1,2,3) <= vec(1,2,3,3));
  check (vec(1,2,3,4) > vec(1,2,3));
  check (vec(1,2,3) == vec(1,2,3));
  check (vec(1,2,3) != vec(1,1,3));
}