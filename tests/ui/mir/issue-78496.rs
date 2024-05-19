//@ run-pass
//@ compile-flags: -Z mir-opt-level=3 -C opt-level=0

// example from #78496
pub enum E<'a> {
    Empty,
    Some(&'a E<'a>),
}

fn f(e: &E) -> u32 {
   if let E::Some(E::Some(_)) = e { 1 } else { 2 }
}

fn main() {
   assert_eq!(f(&E::Empty), 2);
}
