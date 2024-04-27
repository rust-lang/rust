fn main() {
    let mut x: Option<isize> = None;
    match x {
      None => {
          // Note: on this branch, no borrow has occurred.
          x = Some(0);
      }
      Some(ref i) => {
          // But on this branch, `i` is an outstanding borrow
          x = Some(*i+1); //~ ERROR cannot assign to `x` because it is borrowed
          drop(i);
      }
    }
    x.clone(); // just to prevent liveness warnings
}
