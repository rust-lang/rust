fn destructure(x: option<int>) -> int {
    match x {
      none => 0,
      some(ref v) => *v
    }
}

fn main() {
    assert destructure(some(22)) == 22;
}
