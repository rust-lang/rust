#![feature(deref_patterns)]
#![allow(incomplete_features)]

fn main() {
    // FIXME(deref_patterns): fix bindings wrt fake edges
    match vec![1] {
        box [] => unreachable!(),
        box [x] => assert_eq!(x, 1),
        //~^ ERROR used binding isn't initialized
        _ => unreachable!(),
    }
}
