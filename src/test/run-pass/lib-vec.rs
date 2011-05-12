use std;

fn test_init_elt() {
  let vec[uint] v = std::_vec::init_elt[uint](5u, 3u);
  assert (std::_vec::len[uint](v) == 3u);
  assert (v.(0) == 5u);
  assert (v.(1) == 5u);
  assert (v.(2) == 5u);
}

fn id(uint x) -> uint {
  ret x;
}
fn test_init_fn() {
  let fn(uint)->uint op = id;
  let vec[uint] v = std::_vec::init_fn[uint](op, 5u);
  assert (std::_vec::len[uint](v) == 5u);
  assert (v.(0) == 0u);
  assert (v.(1) == 1u);
  assert (v.(2) == 2u);
  assert (v.(3) == 3u);
  assert (v.(4) == 4u);
}

fn test_slice() {
  let vec[int] v = vec(1,2,3,4,5);
  auto v2 = std::_vec::slice[int](v, 2u, 4u);
  assert (std::_vec::len[int](v2) == 2u);
  assert (v2.(0) == 3);
  assert (v2.(1) == 4);
}

fn test_map() {
  fn square(&int x) -> int { ret x * x; }
  let std::option::operator[int, int] op = square;
  let vec[int] v = vec(1, 2, 3, 4, 5);
  let vec[int] s = std::_vec::map[int, int](op, v);
  let int i = 0;
  while (i < 5) {
    assert (v.(i) * v.(i) == s.(i));
    i += 1;
  }
}

fn test_map2() {
  fn times(&int x, &int y) -> int { ret x * y; }
  auto f = times;
  auto v0 = vec(1, 2, 3, 4, 5);
  auto v1 = vec(5, 4, 3, 2, 1);
  auto u = std::_vec::map2[int,int,int](f, v0, v1);

  auto i = 0;
  while (i < 5) {
    assert (v0.(i) * v1.(i) == u.(i));
    i += 1;
  }
}

fn main() {
  test_init_elt();
  test_init_fn();
  test_slice();
  test_map();
  test_map2();
}

