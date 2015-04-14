// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// A specialised string-like structure that is optimised for appending text and
// sequential reading.

// TODO
// Debug
// docs
// char iterator
//   chars -> char_indices and flip order of char/index
// Eq

extern crate unicode;

use std::str::FromStr;
use std::{cmp, fmt};

const MAX_CAPACITY: usize = 0xffff;
const INIT_CAPACITY: usize = 0xff; 

pub struct StringBuffer {
    first: Box<StringNode>,
    // last: &self StringNode
    // Optimisation that saves us from walking the whole list of nodes everytime
    // we append a string.
    last: *mut StringNode,
    // The length of the whole StringBuffer.
    len: usize,
}

pub struct Chars<'a> {
    // Node we're currently iterating over.
    cur_node: &'a StringNode,
    // Byte in cur_node.
    cur_byte: usize,
    // Byte since start of StringBuffer.
    abs_byte: usize,
}

struct StringNode {
    data: String,
    next: Option<Box<StringNode>>,
}

impl StringBuffer {
    pub fn new() -> StringBuffer {
        StringBuffer::with_capacity(INIT_CAPACITY)
    }

    pub fn with_capacity(capacity: usize) -> StringBuffer {
        let mut result = StringBuffer {
            first: box StringNode::with_capacity(capacity),
            last: 0 as *mut StringNode,
            len: 0,
        };
        result.last = &mut *result.first;
        result
    }

    pub fn push_str(&mut self, text: &str) {
        self.len += text.len();
        unsafe {
            // Safety invariant: the `last` field will only ever point to
            // a node owned by self, and will live until destruction of self.
            self.last = (&mut *self.last).push_str(text);
        }
    }

    // Returns the number of characters from start of the last line in the
    // StringBuffer.
    // Note that it is possible for this operation to take a long time in
    // pathological cases (lots of nodes, few line breaks).
    pub fn cur_offset(&self) -> usize {
        unsafe {
            let result = (&*self.last).cur_offset();
            // TODO
            result.unwrap_or_else(|| {
                println!(" FIXME Unimplemented cur_offset across node boundaries");
                0
            })
        }
    }

    pub fn chars<'a>(&'a self) -> Chars<'a> {
        Chars::new(&self.first)
    }
}

impl StringNode {
    fn with_capacity(capacity: usize) -> StringNode {
        StringNode {
            data: String::with_capacity(capacity),
            next: None,
        }
    }

    // Returns a reference to the new last node. 
    fn push_str(&mut self, text: &str) -> &mut StringNode {
        if let Some(ref mut n) = self.next {
            return n.push_str(text);
        }

        if self.data.capacity() - self.data.len() >= text.len() {
            self.data.push_str(text);
            self
        } else {
            self.data.shrink_to_fit();
            let next_cap = cmp::min(cmp::max(self.data.capacity(),
                                             INIT_CAPACITY) * 2,
                                    MAX_CAPACITY);
            let next_cap = cmp::max(next_cap, text.len());
            self.next = Some(box StringNode::with_capacity(next_cap));
            let next = self.next.as_mut().unwrap();
            next.push_str(text);
            &mut **next
        }
    }

    // None if there is no new line in this node.
    fn cur_offset(&self) -> Option<usize> {
        self.data.rfind('\n').map(|i| self.data.len() - i - 1)
    }
}

impl FromStr for StringBuffer {
    type Err = ();
    fn from_str(text: &str) -> Result<StringBuffer, ()> {
        let mut result = StringBuffer::with_capacity(cmp::max(INIT_CAPACITY, text.len()));
        result.push_str(text);
        Ok(result)
    }
}

impl fmt::Display for StringBuffer {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        fn fmt_node(node: &StringNode, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
            try!(write!(fmt, "{}", node.data));
            if let Some(ref n) = node.next {
                fmt_node(n, fmt)
            } else {
                Ok(())
            }
        }

        fmt_node(&self.first, fmt)
    }
}

impl<'a> Iterator for Chars<'a> {
    type Item = (char, usize);

    fn next(&mut self) -> Option<(char, usize)> {
        while self.cur_byte >= self.cur_node.data.len() {
            if let Some(ref n) = self.cur_node.next {
                self.cur_byte = 0;
                self.cur_node = n;
            } else {
                return None;                
            }
        }

        let byte = self.abs_byte;
        let result = self.read_char();

        return Some((result, byte));
    }
}

impl<'a> Chars<'a> {
    fn new<'b>(first_node: &'b StringNode) -> Chars<'b> {
        Chars {
            cur_node: first_node,
            cur_byte: 0,
            abs_byte: 0,
        }
    }

    fn read_char(&mut self) -> char {
        let first_byte = self.read_byte();
        let width = unicode::str::utf8_char_width(first_byte);
        if width == 1 {
            return first_byte as char
        }
        if width == 0 {
            panic!("non-utf8 char in StringBuffer");
        }
        let mut buf = [first_byte, 0, 0, 0];
        {
            let mut start = 1;
            while start < width {
                buf[start] = self.read_byte();
                start += 1;
            }
        }
        match ::std::str::from_utf8(&buf[..width]).ok() {
            Some(s) => s.char_at(0),
            None => panic!("bad chars in StringBuffer")
        }
    }

    fn read_byte(&mut self) -> u8 {
        let result = self.cur_node.data.as_bytes()[self.cur_byte];
        self.cur_byte += 1;
        self.abs_byte += 1;
        result
    }
}


#[cfg(test)]
mod test {
    use super::*;
    // Bug #23157
    use super::{StringNode, INIT_CAPACITY};

    #[test]
    fn test_new() {
        let s = StringBuffer::new();
        assert!(s.len == 0);
        assert!(s.to_string() == "");
        assert!(count_nodes(&s) == 1);
        assert!(first_capacity(&s) == INIT_CAPACITY);

        let s = StringBuffer::with_capacity(64);
        assert!(s.len == 0);
        assert!(s.to_string() == "");
        assert!(count_nodes(&s) == 1);
        assert!(first_capacity(&s) == 64);
    }

    #[test]
    fn test_from_str() {
        let s: StringBuffer = "Hello".parse().unwrap();
        assert!(s.len == 5);
        assert!(s.to_string() == "Hello");
        assert!(count_nodes(&s) == 1);
        assert!(first_capacity(&s) == INIT_CAPACITY);

        let expected = "Hello";
        for ((i, (c, b)), cc) in s.chars().enumerate().zip(expected.chars()) {
            assert!(c == cc);
            assert!(i == b);
        }
    }

    #[test]
    fn test_push_str() {
        let mut s: StringBuffer = "Hello".parse().unwrap();
        assert!(first_capacity(&s) == INIT_CAPACITY);

        s.push_str(" world!");
        assert!(s.to_string() == "Hello world!");
        assert!(s.len == 12);
        s.push_str(" foo");
        assert!(s.to_string() == "Hello world! foo");
        assert!(s.len == 16);

        assert!(count_nodes(&s) == 1);

        let expected = "Hello world! foo";
        for ((i, (c, b)), cc) in s.chars().enumerate().zip(expected.chars()) {
            assert!(c == cc);
            assert!(i == b);
        }
    }

    // push_str requiring multiple nodes
    #[test]
    fn test_push_str_multi() {
        let mut s: StringBuffer = StringBuffer::with_capacity(2);
        assert!(first_capacity(&s) == 2);

        s.push_str("Hello");
        assert!(s.to_string() == "Hello");
        assert!(s.len == 5);
        assert!(count_nodes(&s) == 2);
        s.push_str(" world!");
        assert!(s.to_string() == "Hello world!");
        assert!(s.len == 12);
        assert!(count_nodes(&s) == 2);

        let expected = "Hello world!";
        for ((i, (c, b)), cc) in s.chars().enumerate().zip(expected.chars()) {
            assert!(c == cc);
            assert!(i == b);
        }
    }

    // TODO test unicode

    // Helper methods.
    fn count_nodes(s: &StringBuffer) -> usize {
        count_nodes_from(&s.first)
    }
    fn count_nodes_from(s: &StringNode) -> usize {
        match s.next {
            Some(ref n) => 1 + count_nodes_from(n),
            None => 1,
        }
    }
    fn first_capacity(s: &StringBuffer) -> usize {
        s.first.data.capacity()
    }
}
