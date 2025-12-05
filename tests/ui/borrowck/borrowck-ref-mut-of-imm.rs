fn destructure(x: Option<isize>) -> isize {
    match x {
      None => 0,
      Some(ref mut v) => *v //~ ERROR cannot borrow
    }
}

fn main() {
    assert_eq!(destructure(Some(22)), 22);
}
