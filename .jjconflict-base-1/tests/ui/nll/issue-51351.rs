//
// Regression test for #51351 and #52133: In the case of #51351,
// late-bound regions (like 'a) that were unused within the arguments of
// a function were overlooked and could case an ICE. In the case of #52133,
// LBR defined on the creator function needed to be added to the free regions
// of the closure, as they were not present in the closure's generic
// declarations otherwise.
//
//@ check-pass

fn creash<'a>() {
    let x: &'a () = &();
}

fn produce<'a>() {
   move || {
        let x: &'a () = &();
   };
}

fn main() {}
