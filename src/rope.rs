// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// TODO
// ----
// docs - mod docs, item docs
// tests
// pull out into its own crate
// impl Default, Extend
// impl DoubleEndedIter and ExactSizeIter for RopeChars
// better allocation
// balancing?

extern crate unicode;
use std::fmt;
use std::ops::Range;
use std::num::{SignedInt, Int};

// A Rope, based on an unbalanced binary tree. The rope is somewhat special in
// that it tracks positions in the source text. So when locating a position in
// the rope, the user can use either a current position in the text or a
// position in the source text, which the Rope will adjust to a current position
// whilst searching.
pub struct Rope {
    root: Node,
    len: usize,
    src_len: usize,
    // FIXME: Allocation is very dumb at the moment, we always add another
    // buffer for every inserted string and we never resuse or collect old
    // memory
    storage: Vec<Vec<u8>>
}

// A view over a portion of a Rope. Analagous to string slices (`str`);
pub struct RopeSlice<'rope> {
    // All nodes which make up the slice, in order.
    nodes: Vec<&'rope Lnode>,
    // The offset of the start point in the first node.
    start: usize,
    // The length of text in the last node.
    len: usize,
}

// An iterator over the chars in a rope.
pub struct RopeChars<'rope> {
    data: RopeSlice<'rope>,
    cur_node: usize,
    cur_byte: usize,
    abs_byte: usize,
}

impl Rope {
    // Create an empty rope.
    pub fn new() -> Rope {
        Rope {
            root: Node::empty_inner(),
            len: 0,
            src_len: 0,
            storage: vec![],
        }
    }

    // Uses text as initial storage.
    pub fn from_string(text: String) -> Rope {
        // TODO should split very large texts into segments as we insert

        let mut result = Rope::new();
        result.insert(0, text);
        result.fix_src();
        result
    }

    // When initialising a rope, indicates that the rope is complete wrt the
    // source text.
    fn fix_src(&mut self) {
        self.root.fix_src();
        self.src_len = self.len;
    }

    // Length of the rope.
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn insert_copy(&mut self, start: usize, text: &str) {
        // FIXME If we did clever things with allocation, we could do better here.
        self.insert(start, text.to_string());
    }

    pub fn insert(&mut self, start: usize, text: String) {
        self.insert_inner(start,
                          text,
                          |this, node| this.root.insert(node, start, start))
    }

    pub fn src_insert(&mut self, start: usize, text: String) {
        self.insert_inner(start,
                          text,
                          |this, node| this.root.src_insert(node, start, start))
    }

    fn insert_inner<F>(&mut self,
                       start: usize,
                       text: String,
                       do_insert: F)
        where F: Fn(&mut Rope, Box<Node>) -> NodeAction
    {
        if text.len() == 0 {
            return;
        }

        debug_assert!(start <= self.src_len, "insertion out of bounds of rope");

        let len = text.len();
        let storage = text.into_bytes();
        let new_node = box Node::new_leaf(&storage[..][0] as *const u8, len, 0);
        self.storage.push(storage);

        match do_insert(self, new_node) {
            NodeAction::Change(n, adj) => {
                assert!(adj as usize == len);
                self.root = *n;
            }
            NodeAction::Adjust(adj) => {
                assert!(adj as usize == len);
            }
            _ => panic!("Unexpected action")
        }
        self.len += len;
    }

    pub fn push(&mut self, text: String) {
        let len = self.len();
        self.insert(len, text);
    }

    pub fn push_copy(&mut self, text: &str) {
        // If we did clever things with allocation, we could do better here
        let len = self.len();
        self.insert(len, text.to_string());
    }

    pub fn remove(&mut self, start: usize, end: usize) {
        self.remove_inner(start, end, |this| this.root.remove(start, end, start))
    }

    pub fn src_remove(&mut self, start: usize, end: usize) {
        self.remove_inner(start, end, |this| this.root.src_remove(start, end, start))
    }

    fn remove_inner<F>(&mut self,
                       start: usize,
                       end: usize,
                       do_remove: F)
        where F: Fn(&mut Rope) -> NodeAction
    {
        assert!(end >= start);
        if start == end {
            return;
        }

        match do_remove(self) {
            NodeAction::None => {}
            NodeAction::Remove => {
                self.root = Node::empty_inner();
                self.len = 0;
            }
            NodeAction::Adjust(adj) => self.len = (self.len as isize + adj) as usize,
            NodeAction::Change(node, adj) => {
                self.root = *node;
                self.len = (self.len as isize + adj) as usize;
            }
        }
    }

    // TODO src_replace
    // TODO src_replace_str

    // This can go horribly wrong if you overwrite a grapheme of different size.
    // It is the callers responsibility to ensure that the grapheme at point start
    // has the same size as new_char.
    pub fn replace(&mut self, start: usize, new_char: char) {
        assert!(start + new_char.len_utf8() <= self.len);
        // This is pretty wasteful in that we're allocating for no point, but
        // I think that is better than duplicating a bunch of code.
        // It should be possible to view a &char as a &[u8] somehow, and then
        // we can optimise this (FIXME).
        self.replace_str(start, &new_char.to_string()[..]);
    }

    pub fn replace_str(&mut self, start: usize, new_str: &str) {
        assert!(start + new_str.len() <= self.len);
        self.root.replace(start, new_str);
    }

    // Note, this is not necessarily cheap.
    pub fn col_for_src_loc(&self, src_loc: usize) -> usize {
        assert!(src_loc <= self.src_len);
        match self.root.col_for_src_loc(src_loc) {
            Search::Done(c) | Search::Continue(c) => c
        }
    }

    pub fn slice(&self, Range { start, end }: Range<usize>) -> RopeSlice {
        debug_assert!(end > start && start <= self.len && end <= self.len);
        if start == end {
            return RopeSlice::empty();
        }

        let mut result = RopeSlice::empty();
        self.root.find_slice(start, end, &mut result);
        result
    }

    pub fn full_slice(&self) -> RopeSlice {
        self.slice(0..self.len)
    }

    pub fn src_slice(&self, Range { start, end }: Range<usize>) -> RopeSlice {
        debug_assert!(end > start && start <= self.src_len && end <= self.src_len);
        if start == end {
            return RopeSlice::empty();
        }

        let mut result = RopeSlice::empty();
        self.root.find_src_slice(start, end, &mut result);
        result
    }

    pub fn chars(&self) -> RopeChars {
        RopeChars {
            data: self.full_slice(),
            cur_node: 0,
            cur_byte: 0,
            abs_byte: 0,
        }
    }
}

impl<'rope> RopeSlice<'rope> {
    fn empty<'r>() -> RopeSlice<'r> {
        RopeSlice {
            nodes: vec![],
            start: 0,
            len: 0,
        }
    }
}

impl<'rope> Iterator for RopeChars<'rope> {
    type Item = (char, usize);
    fn next(&mut self) -> Option<(char, usize)> {
        if self.cur_node >= self.data.nodes.len() {
            return None;
        }

        let byte = self.abs_byte;
        let node = self.data.nodes[self.cur_node];
        if self.cur_byte >= node.len {
            self.cur_byte = 0;
            self.cur_node += 1;
            return self.next();
        }

        let result = self.read_char();
        return Some((result, byte));
    }
}

impl<'rope> RopeChars<'rope> {
    fn read_char(&mut self) -> char {
        let first_byte = self.read_byte();
        let width = unicode::str::utf8_char_width(first_byte);
        if width == 1 {
            return first_byte as char
        }
        if width == 0 {
            panic!("non-utf8 char in rope");
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
            None => panic!("bad chars in rope")
        }
    }

    fn read_byte(&mut self) -> u8 {
        let node = self.data.nodes[self.cur_node];
        let addr = node.text as usize + self.cur_byte;
        self.cur_byte += 1;
        self.abs_byte += 1;
        let addr = addr as *const u8;
        unsafe {
            *addr
        }        
    }
}

impl ::std::str::FromStr for Rope {
    type Err = ();
    fn from_str(text: &str) -> Result<Rope, ()> {
        // TODO should split large texts into segments as we insert

        let mut result = Rope::new();
        result.insert_copy(0, text);
        result.fix_src();
        Ok(result)
    }
}

impl<'a> fmt::Display for RopeSlice<'a> {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        if self.nodes.len() == 0 {
            return Ok(());
        }

        let last_idx = self.nodes.len() - 1;
        for (i, n) in self.nodes.iter().enumerate() {
            let mut ptr = n.text;
            let mut len = n.len;
            if i == 0 {
                ptr = (ptr as usize + self.start) as *const u8;
                len -= self.start;
            }
            if i == last_idx {
                len = self.len;
            }
            unsafe {
                try!(write!(fmt,
                            "{}",
                            ::std::str::from_utf8(::std::slice::from_raw_parts(ptr, len)).unwrap()));
            }
        }
        Ok(())
    }
}

impl<'a> fmt::Debug for RopeSlice<'a> {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        let last_idx = self.nodes.len() - 1;
        for (i, n) in self.nodes.iter().enumerate() {
            let mut ptr = n.text;
            let mut len = n.len;
            if i == 0 {
                ptr = (ptr as usize + self.start) as *const u8;
                len -= self.start;
            } else {
                try!(write!(fmt, "|"));
            }
            if i == last_idx {
                len = self.len;
            }
            unsafe {
                try!(write!(fmt,
                            "\"{}\"",
                            ::std::str::from_utf8(::std::slice::from_raw_parts(ptr, len)).unwrap()));
            }
        }
        Ok(())
    }
}

impl fmt::Display for Rope {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(fmt, "{}", self.root)
    }
}

impl fmt::Debug for Rope {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(fmt, "{:?}", self.root)
    }
}

impl fmt::Display for Node {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        match *self {
            Node::InnerNode(Inode { ref left, ref right, .. }) => {
                if let Some(ref left) = *left {
                    write!(fmt, "{}", left)
                } else {
                    Ok(())
                }.and_then(|_| if let Some(ref right) = *right {
                    write!(fmt, "{}", right)
                } else {
                    Ok(())
                })
            }
            Node::LeafNode(Lnode{ ref text, len, .. }) => {
                unsafe {
                    write!(fmt,
                           "{}",
                           ::std::str::from_utf8(::std::slice::from_raw_parts(*text, len)).unwrap())
                }
            }
        }
    }
}

impl fmt::Debug for Node {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        match *self {
            Node::InnerNode(Inode { ref left, ref right, weight, .. }) => {
                try!(write!(fmt, "("));
                if let Some(ref left) = *left {
                    try!(write!(fmt, "left: {:?}", &**left));
                } else {
                    try!(write!(fmt, "left: ()"));
                }
                try!(write!(fmt, ", "));
                if let Some(ref right) = *right {
                    try!(write!(fmt, "right: {:?}", &**right));
                } else {
                    try!(write!(fmt, "right: ()"));
                }
                write!(fmt, "; {})", weight)
            }
            Node::LeafNode(Lnode{ ref text, len, .. }) => {
                unsafe {
                    write!(fmt,
                           "(\"{}\"; {})",
                           ::std::str::from_utf8(::std::slice::from_raw_parts(*text, len)).unwrap(),
                           len)
                }
            }
        }
    }
}

#[derive(Clone, Eq, PartialEq)]
enum Node {
    InnerNode(Inode),
    LeafNode(Lnode),
}

#[derive(Clone, Eq, PartialEq)]
struct Inode {
    weight: usize,
    src_weight: usize,
    left: Option<Box<Node>>,
    right: Option<Box<Node>>,
}

#[derive(Clone, Eq, PartialEq)]
struct Lnode {
    text: *const u8,
    len: usize,
    // text + src_offset = src text (src_offset should always be <= 0)
    src_offset: isize,
}

impl Node {
    fn empty_inner() -> Node {
        Node::InnerNode(Inode {
            left: None,
            right: None,
            weight: 0,
            src_weight: 0,
        })
    }

    fn new_inner(left: Option<Box<Node>>,
                 right: Option<Box<Node>>,
                 weight: usize,
                 src_weight: usize)
    -> Node {
        Node::InnerNode(Inode {
            left: left,
            right: right,
            weight: weight,
            src_weight: src_weight,
        })
    }

    fn new_leaf(text: *const u8, len: usize, src_offset: isize) -> Node {
        Node::LeafNode(Lnode {
            text: text,
            len: len,
            src_offset: src_offset,
        })
    }

    fn len(&self) -> usize {
        match *self {
            Node::InnerNode(Inode { weight, ref right, .. }) => {
                match *right {
                    Some(ref r) => weight + r.len(),
                    None => weight
                }
            }
            Node::LeafNode(Lnode { len, .. }) => len,
        }
    }

    fn fix_src(&mut self) {
        match *self {
            Node::InnerNode(ref mut i) => i.fix_src(),
            Node::LeafNode(ref mut l) => {
                l.src_offset = 0;
            },
        }
    }

    // Most of these methods are just doing dynamic dispatch, TODO use a macro

    // precond: start < end
    fn remove(&mut self, start: usize, end: usize, src_start: usize) -> NodeAction {
        match *self {
            Node::InnerNode(ref mut i) => i.remove(start, end, src_start),
            Node::LeafNode(ref mut l) => l.remove(start, end, src_start),
        }
    }

    fn src_remove(&mut self, start: usize, end: usize, src_start: usize) -> NodeAction {
        match *self {
            Node::InnerNode(ref mut i) => i.src_remove(start, end, src_start),
            Node::LeafNode(ref mut l) => {
                debug!("src_remove: pre-adjust {}-{}; {}", start, end, l.src_offset);
                let start = minz(start as isize + l.src_offset);
                let end = minz(end as isize + l.src_offset);
                let src_start = minz(src_start as isize + l.src_offset);
                debug!("src_remove: post-adjust {}-{}, {}", start, end, src_start);
                if end > start {
                    l.remove(start as usize, end as usize, src_start as usize)
                } else {
                    NodeAction::None
                }
            }
        }
    }

    fn insert(&mut self, node: Box<Node>, start: usize, src_start: usize) -> NodeAction {
        match *self {
            Node::InnerNode(ref mut i) => i.insert(node, start, src_start),
            Node::LeafNode(ref mut l) => l.insert(node, start, src_start),
        }
    }

    fn src_insert(&mut self, node: Box<Node>, start: usize, src_start: usize) -> NodeAction {
        match *self {
            Node::InnerNode(ref mut i) => i.src_insert(node, start, src_start),
            Node::LeafNode(ref mut l) => {
                debug!("src_insert: pre-adjust {}, {}; {}", start, src_start, l.src_offset);
                let start = minz(start as isize + l.src_offset);
                let src_start = minz(src_start as isize + l.src_offset);
                debug!("src_insert: post-adjust {}, {}", start, src_start);
                l.insert(node, start as usize, src_start as usize)
            }
        }
    }

    fn find_slice<'a>(&'a self, start: usize, end: usize, slice: &mut RopeSlice<'a>) {
        match *self {
            Node::InnerNode(ref i) => i.find_slice(start, end, slice),
            Node::LeafNode(ref l) => l.find_slice(start, end, slice),
        }
    }

    fn find_src_slice<'a>(&'a self, start: usize, end: usize, slice: &mut RopeSlice<'a>) {
        match *self {
            Node::InnerNode(ref i) => i.find_src_slice(start, end, slice),
            Node::LeafNode(ref l) => {
                debug!("find_src_slice: pre-adjust {}-{}; {}", start, end, l.src_offset);
                let start = minz(start as isize + l.src_offset);
                let end = minz(end as isize + l.src_offset);
                debug!("find_src_slice: post-adjust {}-{}", start, end);
                if end > start {
                    l.find_slice(start as usize, end as usize, slice);
                }
            }
        }
    }

    fn replace(&mut self, start: usize, new_str: &str) {
        match *self {
            Node::InnerNode(ref mut i) => i.replace(start, new_str),
            Node::LeafNode(ref mut l) => l.replace(start, new_str),
        }        
    }

    fn col_for_src_loc(&self, src_loc: usize) -> Search {
        match *self {
            Node::InnerNode(ref i) => i.col_for_src_loc(src_loc),
            Node::LeafNode(ref l) => l.col_for_src_loc(src_loc),
        }
    }

    fn find_last_char(&self, c: char) -> Option<usize> {
        match *self {
            Node::InnerNode(ref i) => i.find_last_char(c),
            Node::LeafNode(ref l) => l.find_last_char(c),
        }
    }
}

#[derive(Debug, Clone, Eq, PartialEq)]
enum NodeAction {
    None,
    Remove,
    Adjust(isize), // Arg is the length of the old node - the length of the newly adjusted node.
    Change(Box<Node>, isize) // Args are the new node and the change in length.
}

impl Inode {
    fn remove(&mut self, start: usize, end: usize, src_start: usize) -> NodeAction {
        debug!("Inode::remove: {}, {}, {}", start, end, self.weight);

        let left_action = if start <= self.weight {
            if let Some(ref mut left) = self.left {
                left.remove(start, end, src_start)
            } else {
                panic!();
            }
        } else {
            NodeAction::None
        };

        let right_action = if end > self.weight {
            if let Some(ref mut right) = self.right {
                let start = if start < self.weight {
                    0
                } else {
                    start - self.weight
                };
                let src_start = if src_start < self.src_weight {
                    0
                } else {
                    src_start - self.src_weight
                };
                right.remove(start, end - self.weight, src_start)
            } else {
                panic!();
            }
        } else {
            NodeAction::None
        };


        if left_action == NodeAction::Remove && right_action == NodeAction::Remove ||
           left_action == NodeAction::Remove && self.right.is_none() ||
           right_action == NodeAction::Remove && self.left.is_none() {
            return NodeAction::Remove;
        }

        if left_action == NodeAction::Remove {
            return NodeAction::Change(self.right.clone().unwrap(),
                                      -(self.weight as isize));
        }
        if right_action == NodeAction::Remove {
            return NodeAction::Change(self.left.clone().unwrap(),
                                      -(self.right.as_ref().map(|n| n.len()).unwrap() as isize));
        }

        let mut total_adj = 0;
        if let NodeAction::Change(ref n, adj) = left_action {
            self.left = Some(n.clone());
            self.weight = (self.weight as isize + adj) as usize;
            total_adj += adj;
        }
        if let NodeAction::Change(ref n, adj) = right_action {
            self.right = Some(n.clone());
            total_adj += adj;
        }

        if let NodeAction::Adjust(adj) = left_action {
            self.weight = (self.weight as isize + adj) as usize;
            total_adj += adj;
        }
        if let NodeAction::Adjust(adj) = right_action {
            total_adj += adj;
        }

        return NodeAction::Adjust(total_adj);
    }

    fn src_remove(&mut self, start: usize, end: usize, src_start: usize) -> NodeAction {
        // TODO refactor with remove

        debug!("Inode::src_remove: {}, {}, {}/{}", start, end, self.src_weight, self.weight);

        let left_action = if start <= self.src_weight {
            if let Some(ref mut left) = self.left {
                left.src_remove(start, end, src_start)
            } else {
                panic!();
            }
        } else {
            NodeAction::None
        };

        let right_action = if end > self.src_weight {
            if let Some(ref mut right) = self.right {
                let start = if start < self.src_weight {
                    0
                } else {
                    start - self.src_weight
                };
                let src_start = if src_start < self.src_weight {
                    0
                } else {
                    src_start - self.src_weight
                };
                right.src_remove(start, end - self.src_weight, src_start)
            } else {
                panic!();
            }
        } else {
            NodeAction::None
        };


        if left_action == NodeAction::Remove && right_action == NodeAction::Remove ||
           left_action == NodeAction::Remove && self.right.is_none() ||
           right_action == NodeAction::Remove && self.left.is_none() {
            return NodeAction::Remove;
        }

        if left_action == NodeAction::Remove {
            return NodeAction::Change(self.right.clone().unwrap(),
                                      -(self.weight as isize));
        }
        if right_action == NodeAction::Remove {
            return NodeAction::Change(self.left.clone().unwrap(),
                                      -(self.right.as_ref().map(|n| n.len()).unwrap() as isize));
        }

        let mut total_adj = 0;
        if let NodeAction::Change(ref n, adj) = left_action {
            self.left = Some(n.clone());
            self.weight = (self.weight as isize + adj) as usize;
            total_adj += adj;
        }
        if let NodeAction::Change(ref n, adj) = right_action {
            self.right = Some(n.clone());
            total_adj += adj;
        }

        if let NodeAction::Adjust(adj) = left_action {
            self.weight = (self.weight as isize + adj) as usize;
            total_adj += adj;
        }
        if let NodeAction::Adjust(adj) = right_action {
            total_adj += adj;
        }

        return NodeAction::Adjust(total_adj);
    }

    fn insert(&mut self, node: Box<Node>, start: usize, src_start: usize) -> NodeAction {
        let mut total_adj = 0;
        if start <= self.weight {
            let action = if let Some(ref mut left) = self.left {
                left.insert(node, start, src_start)
            } else {
                assert!(self.weight == 0);
                let len = node.len() as isize;
                NodeAction::Change(node, len)
            };

            match action {
                NodeAction::Change(n, adj) => {
                    self.left = Some(n);
                    self.weight += adj as usize;
                    total_adj += adj;
                }
                NodeAction::Adjust(adj) => {
                    self.weight += adj as usize;
                    total_adj += adj;
                }
                _ => panic!("Unexpected action"),
            }
        } else {
            let action = if let Some(ref mut right) = self.right {
                assert!(start >= self.weight);
                assert!(src_start >= self.src_weight);
                right.insert(node, start - self.weight, src_start - self.src_weight)
            } else {
                let len = node.len() as isize;
                NodeAction::Change(node, len)
            };

            match action {
                NodeAction::Change(n, adj) => {
                    self.right = Some(n);
                    total_adj += adj;
                }
                NodeAction::Adjust(adj) => total_adj += adj,
                _ => panic!("Unexpected action"),
            }
        }

        NodeAction::Adjust(total_adj)
    }

    fn src_insert(&mut self, node: Box<Node>, start: usize, src_start: usize) -> NodeAction {
        let mut total_adj = 0;
        if start <= self.src_weight {
            let action = if let Some(ref mut left) = self.left {
                left.src_insert(node, start, src_start)
            } else {
                let len = node.len() as isize;
                NodeAction::Change(node, len)
            };

            match action {
                NodeAction::Change(n, adj) => {
                    self.left = Some(n);
                    self.weight += adj as usize;
                    total_adj += adj;
                }
                NodeAction::Adjust(adj) => {
                    self.weight += adj as usize;
                    total_adj += adj;
                }
                _ => panic!("Unexpected action"),
            }
        } else {
            let action = if let Some(ref mut right) = self.right {
                assert!(start >= self.src_weight);
                assert!(src_start >= self.src_weight);
                right.src_insert(node, start - self.src_weight, src_start - self.src_weight)
            } else {
                let len = node.len() as isize;
                NodeAction::Change(node, len)
            };

            match action {
                NodeAction::Change(n, adj) => {
                    self.right = Some(n);
                    total_adj += adj;
                }
                NodeAction::Adjust(adj) => total_adj += adj,
                _ => panic!("Unexpected action"),
            }
        }

        NodeAction::Adjust(total_adj)
    }

    fn find_slice<'a>(&'a self, start: usize, end: usize, slice: &mut RopeSlice<'a>) {
        debug!("Inode::find_slice: {}, {}, {}", start, end, self.weight);
        if start < self.weight {
            self.left.as_ref().unwrap().find_slice(start, end, slice);
        }
        if end > self.weight {
            let start = if start < self.weight {
                0
            } else {
                start - self.weight
            };
            self.right.as_ref().unwrap().find_slice(start, end - self.weight, slice)
        }
    }

    fn find_src_slice<'a>(&'a self, start: usize, end: usize, slice: &mut RopeSlice<'a>) {
        debug!("Inode::find_src_slice: {}, {}, {}", start, end, self.src_weight);
        if start < self.src_weight && self.left.is_some() {
            self.left.as_ref().unwrap().find_src_slice(start, end, slice);
        }
        if end > self.src_weight && self.right.is_some() {
            let start = if start < self.src_weight {
                0
            } else {
                start - self.src_weight
            };
            self.right.as_ref().unwrap().find_src_slice(start, end - self.src_weight, slice)
        }
    }

    fn replace(&mut self, start: usize, new_str: &str) {
        debug!("Inode::replace: {}, {}, {}", start, new_str, self.weight);
        let end = start + new_str.len();
        if start < self.weight {
            if let Some(ref mut left) = self.left {
                left.replace(start, &new_str[..::std::cmp::min(self.weight-start, new_str.len())]);
            } else {
                panic!();
            }
        }
        if end > self.weight {
            let (start, offset) = if start < self.weight {
                (0, self.weight - start)
            } else {
                (start - self.weight, 0)
            };
            if let Some(ref mut right) = self.right {
                right.replace(start, &new_str[offset..]);
            } else {
                panic!();
            }
        }
    }

    fn fix_src(&mut self) {
        self.src_weight = self.weight;
        if let Some(ref mut left) = self.left {
            left.fix_src();
        }
        if let Some(ref mut right) = self.right {
            right.fix_src();
        }
    }

    fn col_for_src_loc(&self, src_loc: usize) -> Search {
        debug!("Inode::col_for_src_loc: {}, {}", src_loc, self.src_weight);
        let result = if src_loc < self.src_weight {
            if self.left.is_some() {
                Some(self.left.as_ref().unwrap().col_for_src_loc(src_loc))
            } else {
                None
            }
        } else {
            None
        };
        if result.is_none() {
            if self.right.is_some() {
                match self.right.as_ref().unwrap().col_for_src_loc(src_loc - self.src_weight) {
                    Search::Continue(c) if self.left.is_some() => {
                        // TODO broken - need number of chars, not bytes
                        match self.left.as_ref().unwrap().find_last_char('\n') {
                            Some(l) => {
                                Search::Done((self.weight - l - 1) + c)
                            }
                            None => {
                                Search::Continue(c + self.weight)
                            }
                        }
                    }
                    result => result,
                }
            } else {
                panic!("Can't look up source location");
            }
        } else {
            // TODO don't do it this way
            result.unwrap()
        }
    }

    fn find_last_char(&self, c: char) -> Option<usize> {
        // TODO use map or something
        match self.right {
            Some(ref right) => match right.find_last_char(c) {
                Some(x) => return Some(x),
                None => {},
            },
            None => {}
        }
        match self.left {
            Some(ref left) => match left.find_last_char(c) {
                Some(x) => return Some(x),
                None => {},
            },
            None => {}
        }
        None
    }
}

impl Lnode {
    fn remove(&mut self, start: usize, end: usize, src_start: usize) -> NodeAction {
        debug!("Lnode::remove: {}, {}, {}", start, end, self.len);
        assert!(start <= self.len);

        if start == 0 && end >= self.len {
            // The removal span includes us, remove ourselves.
            return NodeAction::Remove;
        }

        let old_len = self.len;
        if start == 0 {
            // Truncate the left of the node.
            self.text = (self.text as usize + end) as *const u8;
            self.len = old_len - end;
            let delta = self.len as isize - old_len as isize;
            self.src_offset += delta;
            return NodeAction::Adjust(delta);
        }

        if end >= self.len {
            // Truncate the right of the node.
            self.len = start;
            return NodeAction::Adjust(self.len as isize - old_len as isize);
        }

        let delta = -((end - start) as isize);
        // Split the node (span to remove is in the middle of the node).
        let new_node = Node::new_inner(
            Some(box Node::new_leaf(self.text, start, self.src_offset)),
            Some(box Node::new_leaf((self.text as usize + end) as *const u8,
                                    old_len - end,
                                    self.src_offset + delta)),
            start,
            src_start);
        return NodeAction::Change(box new_node, delta);
    }

    fn insert(&mut self, mut node: Box<Node>, start: usize, src_start: usize) -> NodeAction {
        match node {
            box Node::LeafNode(ref mut node) => node.src_offset = self.src_offset,
            _ => panic!()
        }

        let len = node.len();
        if start == 0 {
            // Insert at the start of the node
            let new_node = box Node::new_inner(Some(node),
                                               Some(box Node::LeafNode(self.clone())),
                                               len,
                                               0);
            return NodeAction::Change(new_node, len as isize)
        }

        if start == self.len {
            // Insert at the end of the node
            let new_node = box Node::new_inner(Some(box Node::LeafNode(self.clone())),
                                               Some(node),
                                               self.len,
                                               self.len);
            return NodeAction::Change(new_node, len as isize)
        }

        // Insert into the middle of the node
        let left = Some(box Node::new_leaf(self.text, start, self.src_offset));
        let new_left = box Node::new_inner(left, Some(node), start, src_start);
        let right = Some(box Node::new_leaf((self.text as usize + (start)) as *const u8,
                                            self.len - start,
                                            self.src_offset));
        let new_node = box Node::new_inner(Some(new_left), right, start + len, src_start);

        return NodeAction::Change(new_node, len as isize)        
    }

    fn find_slice<'a>(&'a self, start: usize, end: usize, slice: &mut RopeSlice<'a>) {
        debug!("Lnode::find_slice: {}, {}, {}, {}", start, end, self.len, self.src_offset);
        debug_assert!(start < self.len, "Shouldn't have called this fn, we're out of bounds");

        slice.nodes.push(self);
        let mut len = ::std::cmp::min(end, self.len);
        if start > 0 {
            slice.start = start;
            len -= start;
        }
        slice.len = len;
    }

    fn replace(&mut self, start: usize, new_str: &str) {
        debug!("Lnode::replace: {}, {}, {}", start, new_str, self.len);
        debug_assert!(start + new_str.len() <= self.len);
        let addr = (self.text as usize + start) as *mut u8;
        unsafe {
            ::std::intrinsics::volatile_copy_nonoverlapping_memory(addr, &new_str.as_bytes()[0], new_str.len());
        }
    }

    fn col_for_src_loc(&self, src_loc: usize) -> Search {
        debug!("Lnode::col_for_src_loc {}; {}; {}", src_loc, self.len, self.src_offset);
        let loc = if (src_loc as isize) > (self.len as isize - self.src_offset) {
            // The source location we are looking up has been removed
            self.len as isize
        } else {
            (src_loc as isize + self.src_offset) 
        };

        // FIXME if '/n' as u8 is part of a multi-byte grapheme, then this will
        // cause false positives.
        let mut i = loc - 1;
        while i >= 0 {
            unsafe {
                let c = *((self.text as usize + i as usize) as *const u8);
                if c as char == '\n' {
                    debug!("Lnode::col_for_src_loc, return Done({})", loc - i - 1);
                    return Search::Done((loc - i - 1) as usize)
                }
            }
            i -= 1;
        }

        let loc = minz(loc) as usize;
        debug!("Lnode::col_for_src_loc, return Continue({})", loc);
        Search::Continue(loc)
    }

    fn find_last_char(&self, needle: char) -> Option<usize> {
        // FIXME due to multi-byte chars, this will give false positives
        // FIXME use std::str::GraphemeIndices to do this!
        let mut loc = self.len as isize - 1;
        while loc >= 0 {
            unsafe {
                let c = *((self.text as usize + loc as usize) as *const u8);
                if c as char == needle {
                    return Some(loc as usize)
                }
            }
            loc -= 1;
        }

        return None
    }
}

// The state of searching through a rope.
enum Search {
    // TODO comment
    Continue(usize),
    // TODO comment
    Done(usize)
}

fn minz<I: SignedInt>(x: I) -> I {
    if x.is_negative() {
        return I::zero();
    }

    x
}

#[cfg(test)]
mod test {
    use super::*;
    // FIXME is this a Rust bug? Why is minz not imported by the glob import?
    use super::minz;

    #[test]
    fn test_new() {
        let r = Rope::new();
        assert!(r.len() == 0);
        assert!(r.to_string() == "");

        let r = Rope::from_string("Hello world!".to_string());
        assert!(r.len() == 12);
        assert!(r.to_string() == "Hello world!");
    }

    #[test]
    fn test_minz() {
        let x: i32 = 0;
        assert!(super::minz(x) == 0);
        let x: i32 = 42;
        assert!(minz(x) == 42);
        let x: i32 = -42;
        assert!(minz(x) == 0);
        let x: isize = 0;
        assert!(minz(x) == 0);
        let x: isize = 42;
        assert!(minz(x) == 42);
        let x: isize = -42;
        assert!(minz(x) == 0);
    }

    #[test]
    fn test_from_string() {
        let r: Rope = "Hello world!".parse().unwrap();
        assert!(r.to_string() == "Hello world!");
    }

    #[test]
    fn test_remove() {
        let mut r: Rope = "Hello world!".parse().unwrap();
        r.remove(0, 10);
        assert!(r.to_string() == "d!");
        assert!(r.src_slice(0..5).to_string() == "");
        assert!(r.src_slice(10..12).to_string() == "d!");       

        let mut r: Rope = "Hello world!".parse().unwrap();
        r.remove(4, 12);
        assert!(r.to_string() == "Hell");
        // TODO
        //assert!(r.src_slice(0..4).to_string() == "Hell");
        //assert!(r.src_slice(10..12).to_string() == "");       

        let mut r: Rope = "Hello world!".parse().unwrap();
        r.remove(4, 10);
        assert!(r.to_string() == "Helld!");
        // TODO
        //assert!(r.src_slice(1..5).to_string() == "ell");
        assert!(r.src_slice(9..12).to_string() == "d!");
    }

    #[test]
    fn test_insert_copy() {
        let mut r: Rope = "Hello world!".parse().unwrap();
        r.insert_copy(0, "foo");
        assert!(r.to_string() == "fooHello world!");
        assert!(r.slice(2..8).to_string() == "oHello");

        let mut r: Rope = "Hello world!".parse().unwrap();
        r.insert_copy(12, "foo");
        assert!(r.to_string() == "Hello world!foo");
        assert!(r.slice(2..8).to_string() == "llo wo");

        let mut r: Rope = "Hello world!".parse().unwrap();
        r.insert_copy(5, "foo");
        assert!(r.to_string() == "Hellofoo world!");
        assert!(r.slice(2..8).to_string() == "llofoo");
    }

    #[test]
    fn test_push_copy() {
        let mut r: Rope = "Hello world!".parse().unwrap();
        r.push_copy("foo");
        assert!(r.to_string() == "Hello world!foo");
        assert!(r.slice(2..8).to_string() == "llo wo");
    }

    #[test]
    fn test_insert_replace() {
        let mut r: Rope = "hello worl\u{00bb0}!".parse().unwrap();
        r.insert_copy(5, "bb");
        assert!(r.to_string() == "hellobb worlர!");
        r.replace(0, 'H');
        r.replace(15, '~');
        r.replace_str(5, "fo\u{00cb0}");
        assert!(r.to_string() == "Hellofoರrlர~");
        assert!(r.slice(0..10).to_string() == "Hellofoರ");
        assert!(r.slice(5..10).to_string() == "foರ");
        assert!(r.slice(10..15).to_string() == "rlர");

        let expected = "Hellofoರrlர~";
        let mut byte_pos = 0;
        for ((c, b), e) in r.chars().zip(expected.chars()) {
            assert!(c == e);
            assert!(b == byte_pos);
            byte_pos += e.len_utf8();
        }
    }

    #[test]
    fn test_src_insert_remove_col_for_src_loc() {
        let mut r: Rope = "hello\n world!".parse().unwrap();
        r.src_insert(4, "foo".to_string());
        r.src_insert(5, "bar".to_string());
        assert!(r.to_string() == "hellfooobar\n world!");

        r.src_remove(2, 4);
        r.src_remove(10, 12);
        assert!(r.to_string() == "hefooobar\n wor!");

        let expected = "hefooobar\n wor!";
        let mut byte_pos = 0;
        for ((c, b), e) in r.chars().zip(expected.chars()) {
            assert!(c == e);
            assert!(b == byte_pos);
            byte_pos += e.len_utf8();
        }

        let expected = [0, 1, 2, 2, 5, 9, 0, 1, 2, 3, 4, 4, 4];
        for i in 0..13 {
            assert!(r.col_for_src_loc(i) == expected[i]);
        }
    }

    #[test]
    fn test_src_insert() {
        let mut r: Rope = "Hello world!".parse().unwrap();
        r.src_insert(4, "foo".to_string());
        r.src_insert(0, "foo".to_string());
        r.src_insert(12, "foo".to_string());
        assert!(r.to_string() == "fooHellfooo world!foo");
        r.src_insert(4, "bar".to_string());
        r.src_insert(5, "bar".to_string());
        r.src_insert(3, "bar".to_string());
        r.src_insert(0, "bar".to_string());
        r.src_insert(12, "bar".to_string());
        assert!(r.to_string() == "barfooHelbarlbarfooobar world!barfoo");
    }
}
