// Copyright 2012-2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// this tests move up progration, which is not yet implemented

fn foo() -> [u8; 1024] {
        let x = [0; 1024];
        return x;
}

fn main() { }

// END RUST SOURCE
// START rustc.node4.MoveUpPropagation.before.mir
//     bb0: {
//         var0 = [const 0u8; const 1024usize]; // scope 0 at ...
//         tmp0 = var0;                     // scope 1 at ...
//         return = tmp0;                   // scope 1 at ...
//         goto -> bb1;                     // scope 1 at ...
//     }
// END rustc.node4.MoveUpPropagation.before.mir
// START rustc.node4.MoveUpPropagation.after.mir
//     bb0: {
//         return = [const 0u8; const 1024usize]; // scope 0 at return_an_array.rs:2:17: 2:26
//         goto -> bb1;                     // scope 1 at return_an_array.rs:3:13: 3:21
//     }
// END rustc.node4.MoveUpPropagation.after.mir