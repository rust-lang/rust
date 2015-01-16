// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Regression test for #20797.

use std::default::Default;
use std::io::IoResult;
use std::io::fs;
use std::io::fs::PathExtensions;

/// A strategy for acquiring more subpaths to walk.
pub trait Strategy {
  type P: PathExtensions;
  /// Get additional subpaths from a given path.
  fn get_more(&self, item: &Self::P) -> IoResult<Vec<Self::P>>;
  /// Determine whether a path should be walked further.
  /// This is run against each item from `get_more()`.
  fn prune(&self, p: &Self::P) -> bool;
}

/// The basic fully-recursive strategy. Nothing is pruned.
#[derive(Copy, Default)]
pub struct Recursive;

impl Strategy for Recursive {
  type P = Path;
  fn get_more(&self, p: &Path) -> IoResult<Vec<Path>> { fs::readdir(p) }

  fn prune(&self, _: &Path) -> bool { false }
}

/// A directory walker of `P` using strategy `S`.
pub struct Subpaths<S: Strategy> {
    stack: Vec<S::P>,
    strategy: S,
}

impl<S: Strategy> Subpaths<S> {
  /// Create a directory walker with a root path and strategy.
  pub fn new(p: &S::P, strategy: S) -> IoResult<Subpaths<S>> {
    let stack = try!(strategy.get_more(p));
    Ok(Subpaths { stack: stack, strategy: strategy })
  }
}

impl<S: Default + Strategy> Subpaths<S> {
  /// Create a directory walker with a root path and a default strategy.
  pub fn walk(p: &S::P) -> IoResult<Subpaths<S>> {
      Subpaths::new(p, Default::default())
  }
}

impl<S: Default + Strategy> Default for Subpaths<S> {
  fn default() -> Subpaths<S> {
    Subpaths { stack: Vec::new(), strategy: Default::default() }
  }
}

impl<S: Strategy> Iterator for Subpaths<S> {
  type Item = S::P;
  fn next (&mut self) -> Option<S::P> {
    let mut opt_path = self.stack.pop();
    while opt_path.is_some() && self.strategy.prune(opt_path.as_ref().unwrap()) {
      opt_path = self.stack.pop();
    }
    match opt_path {
      Some(path) => {
        if PathExtensions::is_dir(&path) {
          let result = self.strategy.get_more(&path);
          match result {
            Ok(dirs) => { self.stack.extend(dirs.into_iter()); },
            Err(..) => { }
          }
        }
        Some(path)
      }
      None => None,
    }
  }
}

fn main() {
  let mut walker: Subpaths<Recursive> = Subpaths::walk(&Path::new("/home")).unwrap();
}
