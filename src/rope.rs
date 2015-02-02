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
// impl DOubleEndedIter and ExactSizeIter for RopeChars
// better allocation
// balancing
// thread safety/parallisation

extern crate unicode;
use std::fmt;
use std::ops::Range;

// A Rope, based on an unbalanced binary tree.

pub struct Rope {
    root: Node,
    len: usize,
    src_len: usize,
    // FIXME: Allocation is very dumb at the moment, we always add another buffer for every inserted string and we never resuse or collect old memory
    storage: Vec<Vec<u8>>
}

pub struct RopeSlice<'rope> {
    // All nodes which make up the slice, in order.
    nodes: Vec<&'rope Lnode>,
    // The offset of the start point in the first node.
    start: usize,
    // The length of text in the last node.
    len: usize,
}

pub struct RopeChars<'rope> {
    data: RopeSlice<'rope>,
    cur_node: usize,
    cur_byte: usize,
    abs_byte: usize,
}


impl Rope {
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
        // TODO should split large texts into segments as we insert

        let mut result = Rope::new();
        result.insert(0, text);
        result.fix_src();
        result
    }

    fn fix_src(&mut self) {
        self.root.fix_src();
        self.src_len = self.len;
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn insert(&mut self, start: usize, text: String) {
        if text.len() == 0 {
            return;
        }

        debug_assert!(start <= self.len(), "insertion out of bounds of rope");

        let len = text.len();
        let storage = text.into_bytes();
        let new_node = box Node::new_leaf(&storage[][0] as *const u8, len, 0);
        self.storage.push(storage);

        match self.root.insert(new_node, start, start) {
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

    pub fn insert_copy(&mut self, start: usize, text: &str) {
        // If we did clever things with allocation, we could do better here
        self.insert(start, text.to_string());
    }

    pub fn src_insert(&mut self, start: usize, text: String) {
        // TODO refactor with insert
        if text.len() == 0 {
            return;
        }

        debug_assert!(start <= self.src_len, "insertion out of bounds of rope");

        let len = text.len();
        let storage = text.into_bytes();
        let new_node = box Node::new_leaf(&storage[][0] as *const u8, len, 0);
        self.storage.push(storage);

        match self.root.src_insert(new_node, start, start) {
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
        assert!(end >= start);
        if start == end {
            return;
        }

        let action = self.root.remove(start, end, start);
        match action {
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

    pub fn src_remove(&mut self, start: usize, end: usize) {
        // TODO refactor with remove
        assert!(end >= start);
        if start == end {
            return;
        }

        let action = self.root.src_remove(start, end, start);
        match action {
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

    // This can go horribly wrong if you overwrite a grapheme of different size.
    // It is the callers responsibility to ensure that the grapheme at point start
    // has the same size as new_char.
    pub fn replace(&mut self, start: usize, new_char: char) {
        assert!(start + new_char.len_utf8() <= self.len);
        // This is pretty wasteful in that we're allocating for no point, but
        // I think that is better than duplicating a bunch of code.
        // It should be possible to view a &char as a &[u8] somehow, and then
        // we can optimise this (FIXME).
        self.replace_str(start, &new_char.to_string()[]);
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
                            ::std::str::from_utf8(::std::slice::from_raw_buf(&ptr, len)).unwrap()));
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
                            ::std::str::from_utf8(::std::slice::from_raw_buf(&ptr, len)).unwrap()));
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
                           ::std::str::from_utf8(::std::slice::from_raw_buf(text, len)).unwrap())
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
                           ::std::str::from_utf8(::std::slice::from_raw_buf(text, len)).unwrap(),
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

    // All these methods are just doing dynamic dispatch, TODO use a macro

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
                let mut start = start as isize + l.src_offset;
                if start < 0 {
                    start = 0;
                }
                let mut end = end as isize + l.src_offset;
                if end < 0 {
                    end = 0;
                }
                // TODO src_start?
                let mut src_start = src_start as isize + l.src_offset;
                if src_start < 0 {
                    src_start = 0;
                }
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
                let mut start = start as isize + l.src_offset;
                if start < 0 {
                    start = 0;
                }
                // TODO src_start?
                let mut src_start = src_start as isize + l.src_offset;
                if src_start < 0 {
                    src_start = 0;
                }
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
                let mut start = start as isize + l.src_offset;
                if start < 0 {
                    start = 0;
                }
                let mut end = end as isize + l.src_offset;
                if end < 0 {
                    end = 0;
                }
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
            ::std::intrinsics::copy_nonoverlapping_memory(addr, &new_str.as_bytes()[0], new_str.len());
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

        let loc = if loc < 0 {
            0
        } else {
            loc as usize
        };
        debug!("Lnode::col_for_src_loc, return Continue({})", loc);
        Search::Continue(loc)
    }

    fn find_last_char(&self, needle: char) -> Option<usize> {
        // FIXME due to multi-byte chars, this will give false positives
        // I think we must search forwards from the start :-( Perhaps we could
        // track unicode vs ascii or something (I don't think there is an efficient
        // way to read unicode backwards, I might be wrong).
        // std::str::GraphemeIndices can do this!
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

//TODO comment etc.
enum Search {
    Continue(usize),
    Done(usize)
}
