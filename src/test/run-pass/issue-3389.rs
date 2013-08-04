// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

struct trie_node {
    content: ~[~str],
    children: ~[trie_node],
}

fn print_str_vector(vector: ~[~str]) {
    for string in vector.iter() {
        println(*string);
    }
}

pub fn main() {
    let mut node: trie_node = trie_node {
        content: ~[],
        children: ~[]
    };
    let v = ~[~"123", ~"abc"];
    node.content = ~[~"123", ~"abc"];
    print_str_vector(v);
    print_str_vector(node.content.clone());

}
