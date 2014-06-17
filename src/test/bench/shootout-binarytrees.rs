// The Computer Language Benchmarks Game
// http://benchmarksgame.alioth.debian.org/
//
// contributed by the Rust Project Developers

// Copyright (c) 2012-2014 The Rust Project Developers
//
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//
// - Redistributions of source code must retain the above copyright
//   notice, this list of conditions and the following disclaimer.
//
// - Redistributions in binary form must reproduce the above copyright
//   notice, this list of conditions and the following disclaimer in
//   the documentation and/or other materials provided with the
//   distribution.
//
// - Neither the name of "The Computer Language Benchmarks Game" nor
//   the name of "The Computer Language Shootout Benchmarks" nor the
//   names of its contributors may be used to endorse or promote
//   products derived from this software without specific prior
//   written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
// FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
// COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
// INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
// HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
// STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
// OF THE POSSIBILITY OF SUCH DAMAGE.

extern crate arena;

use std::iter::range_step;
use std::sync::Future;
use arena::TypedArena;

enum Tree<'a> {
    Nil,
    Node(&'a Tree<'a>, &'a Tree<'a>, int)
}

fn item_check(t: &Tree) -> int {
    match *t {
        Nil => 0,
        Node(l, r, i) => i + item_check(l) - item_check(r)
    }
}

fn bottom_up_tree<'r>(arena: &'r TypedArena<Tree<'r>>, item: int, depth: int)
                  -> &'r Tree<'r> {
    if depth > 0 {
        arena.alloc(Node(bottom_up_tree(arena, 2 * item - 1, depth - 1),
                         bottom_up_tree(arena, 2 * item, depth - 1),
                         item))
    } else {
        arena.alloc(Nil)
    }
}

fn main() {
    let args = std::os::args();
    let args = args.as_slice();
    let n = if std::os::getenv("RUST_BENCH").is_some() {
        17
    } else if args.len() <= 1u {
        8
    } else {
        from_str(args[1].as_slice()).unwrap()
    };
    let min_depth = 4;
    let max_depth = if min_depth + 2 > n {min_depth + 2} else {n};

    {
        let arena = TypedArena::new();
        let depth = max_depth + 1;
        let tree = bottom_up_tree(&arena, 0, depth);

        println!("stretch tree of depth {}\t check: {}",
                 depth, item_check(tree));
    }

    let long_lived_arena = TypedArena::new();
    let long_lived_tree = bottom_up_tree(&long_lived_arena, 0, max_depth);

    let mut messages = range_step(min_depth, max_depth + 1, 2).map(|depth| {
            use std::num::pow;
            let iterations = pow(2, (max_depth - depth + min_depth) as uint);
            Future::spawn(proc() {
                let mut chk = 0;
                for i in range(1, iterations + 1) {
                    let arena = TypedArena::new();
                    let a = bottom_up_tree(&arena, i, depth);
                    let b = bottom_up_tree(&arena, -i, depth);
                    chk += item_check(a) + item_check(b);
                }
                format!("{}\t trees of depth {}\t check: {}",
                        iterations * 2, depth, chk)
            })
        }).collect::<Vec<Future<String>>>();

    for message in messages.mut_iter() {
        println!("{}", *message.get_ref());
    }

    println!("long lived tree of depth {}\t check: {}",
             max_depth, item_check(long_lived_tree));
}
