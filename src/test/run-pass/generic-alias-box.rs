// xfail-boot
// xfail-stage0

fn id[T](&T t) -> T {
  ret t;
}

fn main() {
  auto expected = @100;
  auto actual = id[@int](expected);
  log *actual;
  check (*expected == *actual);
}