// run-pass

pub fn main() {
    let mut v = Some(22);
    match v {
      None => {}
      Some(ref mut p) => { *p += 1; }
    }
    assert_eq!(v, Some(23));
}
