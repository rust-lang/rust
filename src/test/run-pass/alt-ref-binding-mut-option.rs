fn main() {
    let mut v = some(22);
    match v {
      none => {}
      some(ref mut p) => { *p += 1; }
    }
    assert v == some(23);
}
