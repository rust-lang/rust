// xfail-test

// Broken due to arena API problems.

// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

extern mod std;
use std::arena;

enum tree<'self> {
    nil,
    node(&'self tree<'self>, &'self tree<'self>, int),
}

fn item_check(t: &tree) -> int {
    match *t {
      nil => { return 0; }
      node(left, right, item) => {
        return item + item_check(left) - item_check(right);
      }
    }
}

fn bottom_up_tree<'r>(arena: &'r mut arena::Arena, item: int, depth: int)
                   -> &'r tree<'r> {
    if depth > 0 {
        return arena.alloc(
            || node(bottom_up_tree(arena, 2 * item - 1, depth - 1),
                    bottom_up_tree(arena, 2 * item, depth - 1),
                    item));
    }
    return arena.alloc(|| nil);
}

fn main() {
    let args = os::args();
    let args = if os::getenv(~"RUST_BENCH").is_some() {
        ~[~"", ~"17"]
    } else if args.len() <= 1u {
        ~[~"", ~"8"]
    } else {
        args
    };

    let n = int::from_str(args[1]).get();
    let min_depth = 4;
    let mut max_depth;
    if min_depth + 2 > n {
        max_depth = min_depth + 2;
    } else {
        max_depth = n;
    }

    let mut stretch_arena = arena::Arena();
    let stretch_depth = max_depth + 1;
    let stretch_tree = bottom_up_tree(&mut stretch_arena, 0, stretch_depth);

    io::println(fmt!("stretch tree of depth %d\t check: %d",
                          stretch_depth,
                          item_check(stretch_tree)));

    let mut long_lived_arena = arena::Arena();
    let long_lived_tree = bottom_up_tree(&mut long_lived_arena, 0, max_depth);
    let mut depth = min_depth;
    while depth <= max_depth {
        let iterations = int::pow(2, (max_depth - depth + min_depth) as uint);
        let mut chk = 0;
        let mut i = 1;
        while i <= iterations {
            let mut temp_tree = bottom_up_tree(&mut long_lived_arena, i, depth);
            chk += item_check(temp_tree);
            temp_tree = bottom_up_tree(&mut long_lived_arena, -i, depth);
            chk += item_check(temp_tree);
            i += 1;
        }
        io::println(fmt!("%d\t trees of depth %d\t check: %d",
                         iterations * 2, depth,
                         chk));
        depth += 2;
    }
    io::println(fmt!("long lived trees of depth %d\t check: %d",
                     max_depth,
                     item_check(long_lived_tree)));
}
