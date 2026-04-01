//! Generic-length array strategy.

// Adapted from proptest's array code
// Copyright 2017 Jason Lingle

use core::{marker::PhantomData, mem::MaybeUninit};
use proptest::{
    strategy::{NewTree, Strategy, ValueTree},
    test_runner::TestRunner,
};

#[must_use = "strategies do nothing unless used"]
#[derive(Clone, Copy, Debug)]
pub struct UniformArrayStrategy<S, T> {
    strategy: S,
    _marker: PhantomData<T>,
}

impl<S, T> UniformArrayStrategy<S, T> {
    pub const fn new(strategy: S) -> Self {
        Self {
            strategy,
            _marker: PhantomData,
        }
    }
}

pub struct ArrayValueTree<T> {
    tree: T,
    shrinker: usize,
    last_shrinker: Option<usize>,
}

impl<T, S, const LANES: usize> Strategy for UniformArrayStrategy<S, [T; LANES]>
where
    T: core::fmt::Debug,
    S: Strategy<Value = T>,
{
    type Tree = ArrayValueTree<[S::Tree; LANES]>;
    type Value = [T; LANES];

    fn new_tree(&self, runner: &mut TestRunner) -> NewTree<Self> {
        let tree: [S::Tree; LANES] = unsafe {
            #[allow(clippy::uninit_assumed_init)]
            let mut tree: [MaybeUninit<S::Tree>; LANES] = MaybeUninit::uninit().assume_init();
            for t in tree.iter_mut() {
                *t = MaybeUninit::new(self.strategy.new_tree(runner)?)
            }
            core::mem::transmute_copy(&tree)
        };
        Ok(ArrayValueTree {
            tree,
            shrinker: 0,
            last_shrinker: None,
        })
    }
}

impl<T: ValueTree, const LANES: usize> ValueTree for ArrayValueTree<[T; LANES]> {
    type Value = [T::Value; LANES];

    fn current(&self) -> Self::Value {
        unsafe {
            #[allow(clippy::uninit_assumed_init)]
            let mut value: [MaybeUninit<T::Value>; LANES] = MaybeUninit::uninit().assume_init();
            for (tree_elem, value_elem) in self.tree.iter().zip(value.iter_mut()) {
                *value_elem = MaybeUninit::new(tree_elem.current());
            }
            core::mem::transmute_copy(&value)
        }
    }

    fn simplify(&mut self) -> bool {
        while self.shrinker < LANES {
            if self.tree[self.shrinker].simplify() {
                self.last_shrinker = Some(self.shrinker);
                return true;
            } else {
                self.shrinker += 1;
            }
        }

        false
    }

    fn complicate(&mut self) -> bool {
        if let Some(shrinker) = self.last_shrinker {
            self.shrinker = shrinker;
            if self.tree[shrinker].complicate() {
                true
            } else {
                self.last_shrinker = None;
                false
            }
        } else {
            false
        }
    }
}
