// run-pass
// pretty-expanded FIXME #23616

#![feature(box_syntax)]

fn foo(x: Option<Box<isize>>, b: bool) -> isize {
    match x {
      None => { 1 }
      Some(ref x) if b => { *x.clone() }
      Some(_) => { 0 }
    }
}

pub fn main() {
    foo(Some(box 22), true);
    foo(Some(box 22), false);
    foo(None, true);
    foo(None, false);
}
