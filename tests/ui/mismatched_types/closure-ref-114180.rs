//@ check-fail
#![allow(todo_macro_uses)]

fn main() {
    let mut v = vec![(1,)];
    let compare = |(a,), (e,)| todo!();
    v.sort_by(compare);
    //~^ ERROR type mismatch in closure arguments
}
