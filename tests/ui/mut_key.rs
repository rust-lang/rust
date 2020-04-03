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

#[allow(unused)]
trait Trait {
    type AssociatedType;

    fn trait_fn(&self, set: std::collections::HashSet<Self::AssociatedType>);
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
}
