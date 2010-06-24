fn main() {
  // Testcase for issue #59.
  obj simple(int x, int y) {
    fn sum() -> int {
      ret x + y;
    }
  }

  auto obj0 = simple(1,2);
  auto ctor0 = bind simple(1, _);
  auto ctor1 = bind simple(_, 2);
  auto obj1 = ctor0(2);
  auto obj2 = ctor1(1);
  check (obj0.sum() == 3);
  check (obj1.sum() == 3);
  check (obj2.sum() == 3);
}
