fn destructure(x: Option<int>) -> int {
    match x {
      None => 0,
      Some(ref mut v) => *v //~ ERROR illegal borrow
    }
}

fn main() {
    assert destructure(Some(22)) == 22;
}
