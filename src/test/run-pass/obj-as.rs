
obj big() {
  fn one() -> int { ret 1; }
  fn two() -> int { ret 2; }
  fn three() -> int { ret 3; }
}

type small = obj {
               fn one() -> int;
             };

fn main() {

  let big b = big();
  check (b.one() == 1);
  check (b.two() == 2);
  check (b.three() == 3);

  let small s = b as small;
  check (s.one() == 1);
}