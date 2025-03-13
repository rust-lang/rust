//@compile-flags: --crate-name mut_key
//@check-pass

#![warn(clippy::mutable_key_type)]

use std::cmp::{Eq, PartialEq};
use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::ops::Deref;
use std::sync::atomic::{AtomicUsize, Ordering};

struct Counted<T> {
    count: AtomicUsize,
    val: T,
}

impl<T: Clone> Clone for Counted<T> {
    fn clone(&self) -> Self {
        Self {
            count: AtomicUsize::new(0),
            val: self.val.clone(),
        }
    }
}

impl<T: PartialEq> PartialEq for Counted<T> {
    fn eq(&self, other: &Self) -> bool {
        self.val == other.val
    }
}
impl<T: PartialEq + Eq> Eq for Counted<T> {}

impl<T: Hash> Hash for Counted<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.val.hash(state);
    }
}

impl<T> Deref for Counted<T> {
    type Target = T;

    fn deref(&self) -> &T {
        self.count.fetch_add(1, Ordering::AcqRel);
        &self.val
    }
}

#[derive(Hash, PartialEq, Eq)]
struct ContainsCounted {
    inner: Counted<String>,
}

// This is not linted because `"mut_key::Counted"` is in
// `arc_like_types` in `clippy.toml`
fn should_not_take_this_arg(_v: HashSet<Counted<String>>) {}

fn indirect(_: HashMap<ContainsCounted, usize>) {}

fn main() {
    should_not_take_this_arg(HashSet::new());
    indirect(HashMap::new());
}
