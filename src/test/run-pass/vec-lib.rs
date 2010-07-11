use std;

fn test_init_elt() {
  let vec[uint] v = std._vec.init_elt[uint](uint(5), uint(3));
  check (std._vec.len[uint](v) == uint(3));
  check (v.(0) == uint(5));
  check (v.(1) == uint(5));
  check (v.(2) == uint(5));
}

fn id(uint x) -> uint {
  ret x;
}
fn test_init_fn() {
  let fn(uint)->uint op = id;
  let vec[uint] v = std._vec.init_fn[uint](op, uint(5));
  // FIXME #108: Can't call templated function twice in the same
  // program, at the moment.
  //check (std._vec.len[uint](v) == uint(5));
  check (v.(0) == uint(0));
  check (v.(1) == uint(1));
  check (v.(2) == uint(2));
  check (v.(3) == uint(3));
  check (v.(4) == uint(4));
}

fn main() {
  test_init_elt();
  //XFAIL: test_init_fn();  // Segfaults.
}