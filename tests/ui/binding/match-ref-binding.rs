//@ run-pass

fn destructure(x: Option<isize>) -> isize {
    match x {
      None => 0,
      Some(ref v) => *v
    }
}

pub fn main() {
    assert_eq!(destructure(Some(22)), 22);
}
