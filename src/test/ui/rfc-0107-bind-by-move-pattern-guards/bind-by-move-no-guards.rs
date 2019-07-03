// Adaptation of existing ui test (from way back in
// rust-lang/rust#2329), that starts passing with this feature in
// place.

// build-pass (FIXME(62277): could be check-pass?)

#![feature(bind_by_move_pattern_guards)]

use std::sync::mpsc::channel;

fn main() {
    let (tx, rx) = channel();
    let x = Some(rx);
    tx.send(false);
    match x {
        Some(z) if z.recv().unwrap() => { panic!() },
        Some(z) => { assert!(!z.recv().unwrap()); },
        None => panic!()
    }
}
