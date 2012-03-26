// error-pattern:no public field or method with that name
class cat {
  priv {
    let mutable meows : uint;
  }

  let how_hungry : int;

  new(in_x : uint, in_y : int) { meows = in_x; how_hungry = in_y; }
}

fn main() {
  let nyan : cat = cat(52u, 99);
  assert (nyan.meows == 52u);
}
/*
  other tests:
  not ok to refer to a private method outside a class
  ok to refer to private method within a class
  can't assign to a non-mutable var
  can't assign to a method

  all the same tests, cross-crate
 */