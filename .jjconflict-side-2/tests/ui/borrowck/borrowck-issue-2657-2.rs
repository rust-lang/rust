//@ run-rustfix
fn main() {

    let x: Option<Box<_>> = Some(Box::new(1));

    match x {
      Some(ref y) => {
        let _b = *y; //~ ERROR cannot move out
      }
      _ => {}
    }
}
