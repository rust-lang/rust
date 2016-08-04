// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

struct Baz {
    x: usize,
    y: f32,
    z: bool,
}

fn bar(a: usize) -> Baz {
    Baz { x: a, y: 0.0, z: false }
}

fn main() {}

// END RUST SOURCE
// START rustc.node13.Deaggregator.before.mir
// bb0: {
//     var0 = arg0;                     // scope 0 at main.rs:8:8: 8:9
//     tmp0 = var0;                     // scope 1 at main.rs:9:14: 9:15
//     return = Baz { x: tmp0, y: const F32(0), z: const false }; // scope ...
//     goto -> bb1;                     // scope 1 at main.rs:8:1: 10:2
// }
// END rustc.node13.Deaggregator.before.mir
// START rustc.node13.Deaggregator.after.mir
// bb0: {
//     var0 = arg0;                     // scope 0 at main.rs:8:8: 8:9
//     tmp0 = var0;                     // scope 1 at main.rs:9:14: 9:15
//     (return.0: usize) = tmp0;        // scope 1 at main.rs:9:5: 9:34
//     (return.1: f32) = const F32(0);  // scope 1 at main.rs:9:5: 9:34
//     (return.2: bool) = const false;  // scope 1 at main.rs:9:5: 9:34
//     goto -> bb1;                     // scope 1 at main.rs:8:1: 10:2
// }
// END rustc.node13.Deaggregator.after.mir