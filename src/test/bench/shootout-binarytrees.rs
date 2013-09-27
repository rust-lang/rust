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
use extra::arena::Arena;

enum Tree<'self> {
    Nil,
    Node(&'self Tree<'self>, &'self Tree<'self>, int),
}

fn item_check(t: &Tree) -> int {
    match *t {
      Nil => { return 0; }
      Node(left, right, item) => {
        return item + item_check(left) - item_check(right);
      }
    }
}

fn bottom_up_tree<'r>(arena: &'r Arena, item: int, depth: int)
                   -> &'r Tree<'r> {
    if depth > 0 {
        return arena.alloc(
            || Node(bottom_up_tree(arena, 2 * item - 1, depth - 1),
                    bottom_up_tree(arena, 2 * item, depth - 1),
                    item));
    }
    return arena.alloc(|| Nil);
}

fn main() {
    use std::os;
    use std::int;
    let args = std::os::args();
    let args = if os::getenv("RUST_BENCH").is_some() {
        ~[~"", ~"17"]
    } else if args.len() <= 1u {
        ~[~"", ~"8"]
    } else {
        args
    };

    let n = from_str::<int>(args[1]).unwrap();
    let min_depth = 4;
    let mut max_depth;
    if min_depth + 2 > n {
        max_depth = min_depth + 2;
    } else {
        max_depth = n;
    }

    let stretch_arena = Arena::new();
    let stretch_depth = max_depth + 1;
    let stretch_tree = bottom_up_tree(&stretch_arena, 0, stretch_depth);

    println!("stretch tree of depth {}\t check: {}",
              stretch_depth,
              item_check(stretch_tree));

    let long_lived_arena = Arena::new();
    let long_lived_tree = bottom_up_tree(&long_lived_arena, 0, max_depth);
    let mut depth = min_depth;
    while depth <= max_depth {
        let iterations = int::pow(2, (max_depth - depth + min_depth) as uint);
        let mut chk = 0;
        let mut i = 1;
        while i <= iterations {
            let mut temp_tree = bottom_up_tree(&long_lived_arena, i, depth);
            chk += item_check(temp_tree);
            temp_tree = bottom_up_tree(&long_lived_arena, -i, depth);
            chk += item_check(temp_tree);
            i += 1;
        }
        println!("{}\t trees of depth {}\t check: {}",
                  iterations * 2, depth, chk);
        depth += 2;
    }
    println!("long lived tree of depth {}\t check: {}",
              max_depth,
              item_check(long_lived_tree));
}
