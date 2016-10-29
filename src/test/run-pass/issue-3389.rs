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
    content: Vec<String> ,
    children: Vec<trie_node> ,
}

fn print_str_vector(vector: Vec<String> ) {
    for string in &vector {
        println!("{}", *string);
    }
}

pub fn main() {
    let mut node: trie_node = trie_node {
        content: Vec::new(),
        children: Vec::new()
    };
    let v = vec!["123".to_string(), "abc".to_string()];
    node.content = vec!["123".to_string(), "abc".to_string()];
    print_str_vector(v);
    print_str_vector(node.content.clone());

}
