use std;

fn check_sort(vec[mutable int] v1, vec[mutable int] v2) {
  auto len = std::vec::len[int](v1);

  fn lt(&int a, &int b) -> bool {
    ret a < b;
  }
  fn equal(&int a, &int b) -> bool {
    ret a == b;
  }
  auto f1 = lt;
  auto f2 = equal;
  std::sort::quick_sort3[int](f1, f2, v1);
  auto i = 0u;
  while (i < len) {
    log v2.(i);
    assert (v2.(i) == v1.(i));
    i += 1u;
  }
}


fn main() {
  {
    auto v1 = [mutable 3,7,4,5,2,9,5,8];
    auto v2 = [mutable 2,3,4,5,5,7,8,9];
    check_sort(v1, v2);
  }

  {
    auto v1 = [mutable 1,1,1];
    auto v2 = [mutable 1,1,1];
    check_sort(v1, v2);
  }

  {
    let vec[mutable int] v1 = [mutable];
    let vec[mutable int] v2 = [mutable];
    check_sort(v1, v2);
  }

  {
    auto v1 = [mutable 9];
    auto v2 = [mutable 9];
    check_sort(v1, v2);
  }

  {
    auto v1 = [mutable 9,3,3,3,9];
    auto v2 = [mutable 3,3,3,9,9];
    check_sort(v1, v2);
  }

}

// Local Variables:
// mode: rust;
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// compile-command: "make -k -C .. 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:


