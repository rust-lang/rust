fn plus_one(f: fn() -> int) -> int {
  return f() + 1;
}

fn ret_plus_one() -> fn(fn() -> int) -> int {
  return plus_one;
}

fn main() {
    let z = do (ret_plus_one()) || { 2 };
    assert z == 3;
}
