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
// START rustc.bar.Deaggregator.before.mir
// bb0: {
//     StorageLive(_2);
//     _2 = _1;
//     _0 = Baz::Foo { x: move _2 };
//     StorageDead(_2);
//     return;
// }
// END rustc.bar.Deaggregator.before.mir
// START rustc.bar.Deaggregator.after.mir
// bb0: {
//     StorageLive(_2);
//     _2 = _1;
//     ((_0 as Foo).0: usize) = move _2;
//     discriminant(_0) = 1;
//     StorageDead(_2);
//     return;
// }
// END rustc.bar.Deaggregator.after.mir
