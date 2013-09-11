// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// xfail-fast no compile flags for check-fast

// we want this to be compiled to avoid bitrot, but the actual test
//has to be conducted by a human, i.e. someone (you?) compiling this
//file with a plain rustc invocation and running it and checking it
//works.

// compile-flags: --cfg robot_mode

extern mod extra;
use extra::rl;

static HISTORY_FILE: &'static str = "rl-human-test-history.txt";

fn main() {
    // don't run this in robot mode, but still typecheck it.
    if !cfg!(robot_mode) {
        println("~~ Welcome to the rl test \"suite\". ~~");
        println!("Operations:
 - restrict the history to 2 lines,
 - set the tab-completion to suggest three copies of each of the last 3 letters (or 'empty'),
 - add 'one' and 'two' to the history,
 - save it to `{0}`,
 - add 'three',
 - prompt & save input (check the history & completion work and contains only 'two', 'three'),
 - load from `{0}`
 - prompt & save input (history should be 'one', 'two' again),
 - prompt once more.

The bool return values of each step are printed.",
                 HISTORY_FILE);

        println!("restricting history length: {}", rl::set_history_max_len(3));

        do rl::complete |line, suggest| {
            if line.is_empty() {
                suggest(~"empty")
            } else {
                for c in line.rev_iter().take(3) {
                    suggest(format!("{0}{1}{1}{1}", line, c))
                }
            }
        }

        println!("adding 'one': {}", rl::add_history("one"));
        println!("adding 'two': {}", rl::add_history("two"));

        println!("saving history: {}", rl::save_history(HISTORY_FILE));

        println!("adding 'three': {}", rl::add_history("three"));

        match rl::read("> ") {
            Some(s) => println!("saving input: {}", rl::add_history(s)),
            None => return
        }
        println!("loading history: {}", rl::load_history(HISTORY_FILE));

        match rl::read("> ") {
            Some(s) => println!("saving input: {}", rl::add_history(s)),
            None => return
        }

        rl::read("> ");
    }
}
