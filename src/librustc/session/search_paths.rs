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
use std::path::{Path, PathBuf};
use session::early_error;
use syntax::diagnostic;

#[derive(Clone, Debug)]
pub struct SearchPaths {
    paths: Vec<(PathKind, PathBuf)>,
}

pub struct Iter<'a> {
    kind: PathKind,
    iter: slice::Iter<'a, (PathKind, PathBuf)>,
}

#[derive(Eq, PartialEq, Clone, Copy, Debug)]
pub enum PathKind {
    Native,
    Crate,
    Dependency,
    Framework,
    ExternFlag,
    All,
}

impl SearchPaths {
    pub fn new() -> SearchPaths {
        SearchPaths { paths: Vec::new() }
    }

    pub fn add_path(&mut self, path: &str, cfg: diagnostic::EmitterConfig) {
        let (kind, path) = if path.starts_with("native=") {
            (PathKind::Native, &path["native=".len()..])
        } else if path.starts_with("crate=") {
            (PathKind::Crate, &path["crate=".len()..])
        } else if path.starts_with("dependency=") {
            (PathKind::Dependency, &path["dependency=".len()..])
        } else if path.starts_with("framework=") {
            (PathKind::Framework, &path["framework=".len()..])
        } else if path.starts_with("all=") {
            (PathKind::All, &path["all=".len()..])
        } else {
            (PathKind::All, path)
        };
        if path.is_empty() {
            early_error(cfg, "empty search path given via `-L`");
        }
        self.paths.push((kind, PathBuf::from(path)));
    }

    pub fn iter(&self, kind: PathKind) -> Iter {
        Iter { kind: kind, iter: self.paths.iter() }
    }
}

impl<'a> Iterator for Iter<'a> {
    type Item = (&'a Path, PathKind);

    fn next(&mut self) -> Option<(&'a Path, PathKind)> {
        loop {
            match self.iter.next() {
                Some(&(kind, ref p)) if self.kind == PathKind::All ||
                                        kind == PathKind::All ||
                                        kind == self.kind => {
                    return Some((p, kind))
                }
                Some(..) => {}
                None => return None,
            }
        }
    }
}
