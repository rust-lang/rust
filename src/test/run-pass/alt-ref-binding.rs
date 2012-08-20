fn destructure(x: Option<int>) -> int {
    match x {
      None => 0,
      Some(ref v) => *v
    }
}

fn main() {
    assert destructure(Some(22)) == 22;
}
