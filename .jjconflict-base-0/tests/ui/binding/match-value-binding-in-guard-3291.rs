//@ run-pass

fn foo(x: Option<Box<isize>>, b: bool) -> isize {
    match x {
      None => { 1 }
      Some(ref x) if b => { *x.clone() }
      Some(_) => { 0 }
    }
}

pub fn main() {
    foo(Some(Box::new(22)), true);
    foo(Some(Box::new(22)), false);
    foo(None, true);
    foo(None, false);
}
