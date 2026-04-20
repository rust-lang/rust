//@ check-fail
#![allow(todo_macro_calls)]

fn main() {
    let mut v = vec![(1,)];
    let compare = |(a,), (e,)| todo!();
    v.sort_by(compare);
    //~^ ERROR type mismatch in closure arguments
}
