use std::cell::Cell;
use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};

struct Key(AtomicUsize);

impl Clone for Key {
    fn clone(&self) -> Self {
        Key(AtomicUsize::new(self.0.load(Relaxed)))
    }
}

impl PartialEq for Key {
    fn eq(&self, other: &Self) -> bool {
        self.0.load(Relaxed) == other.0.load(Relaxed)
    }
}

impl Eq for Key {}

impl Hash for Key {
    fn hash<H: Hasher>(&self, h: &mut H) {
        self.0.load(Relaxed).hash(h);
    }
}

fn should_not_take_this_arg(m: &mut HashMap<Key, usize>, _n: usize) -> HashSet<Key> {
    let _other: HashMap<Key, bool> = HashMap::new();
    m.keys().cloned().collect()
}

fn this_is_ok(_m: &mut HashMap<usize, Key>) {}

// Raw pointers are hashed by the address they point to, so it doesn't matter if they point to a
// type with interior mutability.  See:
// - clippy issue: https://github.com/rust-lang/rust-clippy/issues/6745
// - std lib: https://github.com/rust-lang/rust/blob/1.54.0/library/core/src/hash/mod.rs#L717-L736
// So these are OK:
fn raw_ptr_is_ok(_m: &mut HashMap<*const Key, ()>) {}
fn raw_mut_ptr_is_ok(_m: &mut HashMap<*mut Key, ()>) {}

#[allow(unused)]
trait Trait {
    type AssociatedType;

    fn trait_fn(&self, set: HashSet<Self::AssociatedType>);
}

fn generics_are_ok_too<K>(_m: &mut HashSet<K>) {
    // nothing to see here, move along
}

fn tuples<U>(_m: &mut HashMap<((), U), ()>) {}

fn tuples_bad<U>(_m: &mut HashMap<(Key, U), bool>) {}

fn main() {
    let _ = should_not_take_this_arg(&mut HashMap::new(), 1);
    this_is_ok(&mut HashMap::new());
    tuples::<Key>(&mut HashMap::new());
    tuples::<()>(&mut HashMap::new());
    tuples_bad::<()>(&mut HashMap::new());

    raw_ptr_is_ok(&mut HashMap::new());
    raw_mut_ptr_is_ok(&mut HashMap::new());
}
