//@ build-pass

// Regression test for #20797.

use std::default::Default;
use std::io;
use std::fs;
use std::path::PathBuf;

pub trait PathExtensions {
    fn is_dir(&self) -> bool { false }
}

impl PathExtensions for PathBuf {}

/// A strategy for acquiring more subpaths to walk.
pub trait Strategy {
    type P: PathExtensions;
    /// Gets additional subpaths from a given path.
    fn get_more(&self, item: &Self::P) -> io::Result<Vec<Self::P>>;
    /// Determine whether a path should be walked further.
    /// This is run against each item from `get_more()`.
    fn prune(&self, p: &Self::P) -> bool;
}

/// The basic fully-recursive strategy. Nothing is pruned.
#[derive(Copy, Clone, Default)]
pub struct Recursive;

impl Strategy for Recursive {
    type P = PathBuf;
    fn get_more(&self, p: &PathBuf) -> io::Result<Vec<PathBuf>> {
        Ok(fs::read_dir(p).unwrap().map(|s| s.unwrap().path()).collect())
    }

    fn prune(&self, _: &PathBuf) -> bool { false }
}

/// A directory walker of `P` using strategy `S`.
pub struct Subpaths<S: Strategy> {
    stack: Vec<S::P>,
    strategy: S,
}

impl<S: Strategy> Subpaths<S> {
    /// Creates a directory walker with a root path and strategy.
    pub fn new(p: &S::P, strategy: S) -> io::Result<Subpaths<S>> {
        let stack = strategy.get_more(p)?;
        Ok(Subpaths { stack: stack, strategy: strategy })
    }
}

impl<S: Default + Strategy> Subpaths<S> {
    /// Creates a directory walker with a root path and a default strategy.
    pub fn walk(p: &S::P) -> io::Result<Subpaths<S>> {
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
                if path.is_dir() {
                    let result = self.strategy.get_more(&path);
                    match result {
                        Ok(dirs) => { self.stack.extend(dirs); },
                        Err(..) => { }
                    }
                }
                Some(path)
            }
            None => None,
        }
    }
}

fn _foo() {
    let _walker: Subpaths<Recursive> = Subpaths::walk(&PathBuf::from("/home")).unwrap();
}

fn main() {}
