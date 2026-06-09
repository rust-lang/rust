//@ run-pass

pub fn main() {
    let mut x = None;
    match x {
      None => {
        // It is ok to reassign x here, because there is in
        // fact no outstanding loan of x!
        x = Some(0);
      }
      Some(_) => { }
    }
    assert_eq!(x, Some(0));
}
