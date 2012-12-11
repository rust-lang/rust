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
    mut meow: fn@(),
}

fn cat() -> cat {
    cat {
        meow: fn@() { error!("meow"); }
    }
}

type kitty_info = {kitty: cat};

// Code compiles and runs successfully if we add a + before the first arg
fn nyan(kitty: cat, _kitty_info: kitty_info) {
    (kitty.meow)();
}

fn main() {
    let mut kitty = cat();
    nyan(copy kitty, {kitty: copy kitty});
}
