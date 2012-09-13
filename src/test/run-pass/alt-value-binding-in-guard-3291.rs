fn foo(x: Option<~int>, b: bool) -> int {
    match x {
      None => { 1 }
      Some(copy x) if b => { *x }
      Some(_) => { 0 }
    }
}

fn main() {
    foo(Some(~22), true);
    foo(Some(~22), false);
    foo(None, true);
    foo(None, false);
}