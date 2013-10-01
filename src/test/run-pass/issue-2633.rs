// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

struct cat {
    meow: extern "Rust" fn(),
}

fn meow() {
    error2!("meow")
}

fn cat() -> cat {
    cat {
        meow: meow,
    }
}

struct KittyInfo {kitty: cat}

// Code compiles and runs successfully if we add a + before the first arg
fn nyan(kitty: cat, _kitty_info: KittyInfo) {
    (kitty.meow)();
}

pub fn main() {
    let kitty = cat();
    nyan(kitty, KittyInfo {kitty: kitty});
}
