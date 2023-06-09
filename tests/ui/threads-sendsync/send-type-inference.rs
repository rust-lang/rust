// run-pass
#![allow(unused_must_use)]
#![allow(dead_code)]
#![allow(unused_mut)]
// pretty-expanded FIXME #23616

use std::sync::mpsc::{channel, Sender};

// tests that ctrl's type gets inferred properly
struct Command<K, V> {
    key: K,
    val: V
}

fn cache_server<K:Send+'static,V:Send+'static>(mut tx: Sender<Sender<Command<K, V>>>) {
    let (tx1, _rx) = channel();
    tx.send(tx1);
}
pub fn main() { }
