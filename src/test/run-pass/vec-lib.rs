use std;

fn test_init_elt() {
  let vec[uint] v = std._vec.init_elt[uint](5 as uint, 3 as uint);
  check (std._vec.len[uint](v) == (3 as uint));
  check (v.(0) == (5 as uint));
  check (v.(1) == (5 as uint));
  check (v.(2) == (5 as uint));
}

fn id(uint x) -> uint {
  ret x;
}
fn test_init_fn() {
  let fn(uint)->uint op = id;
  let vec[uint] v = std._vec.init_fn[uint](op, (5 as uint));
  check (std._vec.len[uint](v) == (5 as uint));
  check (v.(0) == (0 as uint));
  check (v.(1) == (1 as uint));
  check (v.(2) == (2 as uint));
  check (v.(3) == (3 as uint));
  check (v.(4) == (4 as uint));
}

fn test_slice() {
  let vec[int] v = vec(1,2,3,4,5);
  auto v2 = std._vec.slice[int](v, 2, 4);
  check (std._vec.len[int](v2) == (2 as uint));
  check (v2.(0) == 3);
  check (v2.(1) == 4);
}

fn test_map() {
  fn square(&int x) -> int { ret x * x; }
  let std.util.operator[int, int] op = square;
  let vec[int] v = vec(1, 2, 3, 4, 5);
  let vec[int] s = std._vec.map[int, int](op, v);
  let int i = 0;
  while (i < 5) {
    check (v.(i) == s.(i));
    i += 1;
  }
}

fn main() {
  test_init_elt();
  //XFAIL: test_init_fn();  // Segfaults.
  test_slice();
}
