//@ check-pass
//@ compile-flags: --test


#![deny(dead_code)]

mod m {
    pub fn other() {}
}

use m::other as main;
