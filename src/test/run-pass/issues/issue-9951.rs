// run-pass
// pretty-expanded FIXME #23616

#![allow(unused_variables)]

trait Bar {
  fn noop(&self);
}
impl Bar for u8 {
  fn noop(&self) {}
}

fn main() {
    let (a, b) = (&5u8 as &Bar, &9u8 as &Bar);
    let (c, d): (&Bar, &Bar) = (a, b);

    let (a, b) = (Box::new(5u8) as Box<Bar>, Box::new(9u8) as Box<Bar>);
    let (c, d): (&Bar, &Bar) = (&*a, &*b);

    let (c, d): (&Bar, &Bar) = (&5, &9);
}
