// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::sync::mpsc::{channel, Sender};
use std::thread::Thread;

struct complainer {
    tx: Sender<bool>,
}

impl Drop for complainer {
    fn drop(&mut self) {
        println!("About to send!");
        self.tx.send(true).unwrap();
        println!("Sent!");
    }
}

fn complainer(tx: Sender<bool>) -> complainer {
    println!("Hello!");
    complainer {
        tx: tx
    }
}

fn f(tx: Sender<bool>) {
    let _tx = complainer(tx);
    panic!();
}

pub fn main() {
    let (tx, rx) = channel();
    let _t = Thread::scoped(move|| f(tx.clone()));
    println!("hiiiiiiiii");
    assert!(rx.recv().unwrap());
}
