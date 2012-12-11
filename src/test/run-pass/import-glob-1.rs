// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use a1::b1::word_traveler;

mod a1 {
    #[legacy_exports];
    //
    mod b1 {
        #[legacy_exports];
        //
        use a2::b1::*;
        //         = move\
        export word_traveler; //           |
    }
    //           |
    mod b2 {
        #[legacy_exports];
        //           |
        use a2::b2::*;
        // = move\  -\   |
        export word_traveler; //   |   |   |
    } //   |   |   |
}
//   |   |   |
//   |   |   |
mod a2 {
    #[legacy_exports];
    //   |   |   |
    #[abi = "cdecl"]
    #[nolink]
    extern mod b1 {
        #[legacy_exports];
        //   |   |   |
        use a1::b2::*;
        //   | = move/  -/
        export word_traveler; //   |
    }
    //   |
    mod b2 {
        #[legacy_exports];
        //   |
        fn word_traveler() { //   |
            debug!("ahoy!"); //  -/
        } //
    } //
}
//


fn main() { word_traveler(); }
