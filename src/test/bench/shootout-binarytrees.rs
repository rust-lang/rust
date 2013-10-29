// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

extern mod extra;

use std::iter::range_step;
use extra::arena::Arena;
use extra::future::Future;

enum Tree<'self> {
    Nil,
    Node(&'self Tree<'self>, &'self Tree<'self>, int)
}

fn item_check(t: &Tree) -> int {
    match *t {
        Nil => 0,
        Node(l, r, i) => i + item_check(l) - item_check(r)
    }
}

fn bottom_up_tree<'r>(arena: &'r Arena, item: int, depth: int) -> &'r Tree<'r> {
    if depth > 0 {
        do arena.alloc {
            Node(bottom_up_tree(arena, 2 * item - 1, depth - 1),
                 bottom_up_tree(arena, 2 * item, depth - 1),
                 item)
        }
    } else {arena.alloc(|| Nil)}
}

fn main() {
    let args = std::os::args();
    let n = if std::os::getenv("RUST_BENCH").is_some() {
        17
    } else if args.len() <= 1u {
        8
    } else {
        from_str(args[1]).unwrap()
    };
    let min_depth = 4;
    let max_depth = if min_depth + 2 > n {min_depth + 2} else {n};

    {
        let arena = Arena::new();
        let depth = max_depth + 1;
        let tree = bottom_up_tree(&arena, 0, depth);

        println!("stretch tree of depth {}\t check: {}",
                 depth, item_check(tree));
    }

    let long_lived_arena = Arena::new();
    let long_lived_tree = bottom_up_tree(&long_lived_arena, 0, max_depth);

    let mut messages = range_step(min_depth, max_depth + 1, 2).map(|depth| {
            use std::int::pow;
            let iterations = pow(2, (max_depth - depth + min_depth) as uint);
            do Future::spawn {
                let mut chk = 0;
                for i in range(1, iterations + 1) {
                    let arena = Arena::new();
                    let a = bottom_up_tree(&arena, i, depth);
                    let b = bottom_up_tree(&arena, -i, depth);
                    chk += item_check(a) + item_check(b);
                }
                format!("{}\t trees of depth {}\t check: {}",
                        iterations * 2, depth, chk)
            }
        }).to_owned_vec();

    for message in messages.mut_iter() {
        println(*message.get_ref());
    }

    println!("long lived tree of depth {}\t check: {}",
             max_depth, item_check(long_lived_tree));
}
