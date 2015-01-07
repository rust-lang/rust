// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::slice;

#[derive(Clone)]
pub struct SearchPaths {
    paths: Vec<(PathKind, Path)>,
}

pub struct Iter<'a> {
    kind: PathKind,
    iter: slice::Iter<'a, (PathKind, Path)>,
}

#[derive(Eq, PartialEq, Clone, Copy)]
pub enum PathKind {
    Native,
    Crate,
    Dependency,
    All,
}

impl SearchPaths {
    pub fn new() -> SearchPaths {
        SearchPaths { paths: Vec::new() }
    }

    pub fn add_path(&mut self, path: &str) {
        let (kind, path) = if path.starts_with("native=") {
            (PathKind::Native, path.slice_from("native=".len()))
        } else if path.starts_with("crate=") {
            (PathKind::Crate, path.slice_from("crate=".len()))
        } else if path.starts_with("dependency=") {
            (PathKind::Dependency, path.slice_from("dependency=".len()))
        } else if path.starts_with("all=") {
            (PathKind::All, path.slice_from("all=".len()))
        } else {
            (PathKind::All, path)
        };
        self.paths.push((kind, Path::new(path)));
    }

    pub fn iter(&self, kind: PathKind) -> Iter {
        Iter { kind: kind, iter: self.paths.iter() }
    }
}

impl<'a> Iterator for Iter<'a> {
    type Item = &'a Path;

    fn next(&mut self) -> Option<&'a Path> {
        loop {
            match self.iter.next() {
                Some(&(kind, ref p)) if self.kind == PathKind::All ||
                                        kind == PathKind::All ||
                                        kind == self.kind => return Some(p),
                Some(..) => {}
                None => return None,
            }
        }
    }
}
