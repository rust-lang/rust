//@ check-pass
fn destructure(x: Option<isize>) -> isize {
    match x {
      None => 0,
      Some(ref mut v) => *v //~ WARNING cannot borrow
    }
}

fn main() {
    assert_eq!(destructure(Some(22)), 22);
}
