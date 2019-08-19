// run-pass
#![allow(non_camel_case_types)]

// ignore-emscripten no threads support

/*
  Make sure we can spawn tasks that take different types of
  parameters. This is based on a test case for #520 provided by Rob
  Arnold.
 */

use std::thread;
use std::sync::mpsc::{channel, Sender};

type ctx = Sender<isize>;

fn iotask(_tx: &ctx, ip: String) {
    assert_eq!(ip, "localhost".to_string());
}

pub fn main() {
    let (tx, _rx) = channel::<isize>();
    let t = thread::spawn(move|| iotask(&tx, "localhost".to_string()) );
    t.join().ok().unwrap();
}
