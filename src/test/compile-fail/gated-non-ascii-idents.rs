// Copyright 2013-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-fast feature doesn't work.

#[feature(struct_variant)];

extern mod bäz; //~ ERROR non-ascii idents

use föö::bar; //~ ERROR non-ascii idents

mod föö { //~ ERROR non-ascii idents
    pub fn bar() {}
}

fn bär( //~ ERROR non-ascii idents
    bäz: int //~ ERROR non-ascii idents
    ) {
    let _ö: int; //~ ERROR non-ascii idents

    match (1, 2) {
        (_ä, _) => {} //~ ERROR non-ascii idents
    }
}

struct Föö { //~ ERROR non-ascii idents
    föö: int //~ ERROR non-ascii idents
}

enum Bär { //~ ERROR non-ascii idents
    Bäz { //~ ERROR non-ascii idents
        qüx: int //~ ERROR non-ascii idents
    }
}

extern {
    fn qüx();  //~ ERROR non-ascii idents
}

fn main() {}
