fn destructure(x: option<int>) -> int {
    match x {
      none => 0,
      some(ref mut v) => *v //~ ERROR illegal borrow
    }
}

fn main() {
    assert destructure(some(22)) == 22;
}
