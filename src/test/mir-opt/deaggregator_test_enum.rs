// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

enum Baz {
    Empty,
    Foo { x: usize },
}

fn bar(a: usize) -> Baz {
    Baz::Foo { x: a }
}

fn main() {
    let x = bar(10);
    match x {
        Baz::Empty => println!("empty"),
        Baz::Foo { x } => println!("{}", x),
    };
}

// END RUST SOURCE
// START rustc.node10.Deaggregator.before.mir
// bb0: {
//     var0 = arg0;                     // scope 0 at main.rs:7:8: 7:9
//     tmp0 = var0;                     // scope 1 at main.rs:8:19: 8:20
//     return = Baz::Foo { x: tmp0 };   // scope 1 at main.rs:8:5: 8:21
//     goto -> bb1;                     // scope 1 at main.rs:7:1: 9:2
// }
// END rustc.node10.Deaggregator.before.mir
// START rustc.node10.Deaggregator.after.mir
// bb0: {
//     var0 = arg0;                     // scope 0 at main.rs:7:8: 7:9
//     tmp0 = var0;                     // scope 1 at main.rs:8:19: 8:20
//     ((return as Foo).0: usize) = tmp0; // scope 1 at main.rs:8:5: 8:21
//     discriminant(return) = 1;         // scope 1 at main.rs:8:5: 8:21
//     goto -> bb1;                     // scope 1 at main.rs:7:1: 9:2
// }
// END rustc.node10.Deaggregator.after.mir