//@ run-pass
#![allow(unused_variables)]

use std::sync::atomic::{AtomicBool, Ordering};

static DROP: AtomicBool = AtomicBool::new(false);

struct ConnWrap(Conn);
impl ::std::ops::Deref for ConnWrap {
    type Target=Conn;
    fn deref(&self) -> &Conn { &self.0 }
}

struct Conn;
impl Drop for  Conn {
    fn drop(&mut self) {
        DROP.store(true, Ordering::Relaxed);
    }
}

fn inner() {
    let conn = &*match Some(ConnWrap(Conn)) {
        Some(val) => val,
        None => return,
    };
    return;
}

fn main() {
    inner();
    assert_eq!(DROP.load(Ordering::Relaxed), true);
}
