//! Repro for a caught-panic corruption path in `std::collections::HashMap`.
//!
//! The bug class is: start a multi-step internal transition, let user code
//! panic in the middle, catch the unwind, and keep using the partially updated
//! object.
//!
//! In this case the user-controlled hook is `BuildHasher::build_hasher` during
//! an in-place rehash. The table keeps its logical length after the panic, but
//! lookups can no longer find the original keys and iteration starts yielding
//! repeated garbage-like entries.

use hashbrown::HashMap;
use std::collections::BTreeSet;
use std::{
    hash::{BuildHasher, Hash, Hasher},
    panic::{AssertUnwindSafe, catch_unwind},
    sync::Mutex,
    sync::atomic::{AtomicUsize, Ordering},
};

/// One-shot panic switch used to trigger the first `build_hasher` call that
/// occurs inside `reserve(1)`.
static PANIC_COUNTER: AtomicUsize = AtomicUsize::new(0);
static TEST_LOCK: Mutex<()> = Mutex::new(());

/// A deterministic hasher that maps everything to the same bucket group.
///
/// This maximizes collisions and makes the in-place rehash path easy to reach
/// with a small, fixed workload.
#[derive(Default)]
struct ZeroHasher;

impl Hasher for ZeroHasher {
    fn finish(&self) -> u64 {
        0
    }

    fn write(&mut self, _bytes: &[u8]) {}
}

/// `BuildHasher` that panics once when armed.
///
/// Using a panicking build hook mirrors the upstream interner trigger more
/// closely than a panicking `Hash` impl.
#[derive(Clone, Default)]
struct PanicBuildHasher;

impl BuildHasher for PanicBuildHasher {
    type Hasher = ZeroHasher;

    fn build_hasher(&self) -> Self::Hasher {
        if PANIC_COUNTER.fetch_sub(1, Ordering::SeqCst) == 0 {
            panic!("panic in BuildHasher::build_hasher");
        }
        ZeroHasher
    }
}

/// Simple integer key type so the test can verify reachability after the panic.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
struct Key(u64);

type Map = HashMap<Key, u64, PanicBuildHasher>;

/// Fill a map until `len == capacity`.
///
/// With the current toolchain this yields a map with `len == capacity == 224`
/// when constructed from `with_capacity_and_hasher(128, ...)`.
fn make_full_map() -> Map {
    PANIC_COUNTER.store(!0, Ordering::SeqCst);
    let mut map = HashMap::with_capacity_and_hasher(128, PanicBuildHasher);
    for i in 0.. {
        map.insert(Key(i), i);
        if map.len() == map.capacity() {
            return map;
        }
    }
    unreachable!()
}

fn panics_silently(f: impl FnOnce()) -> bool {
    let previous_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let panicked = catch_unwind(AssertUnwindSafe(f)).is_err();
    std::panic::set_hook(previous_hook);
    panicked
}

fn hashmap_reserve_survives_panicking_build_hasher_inner(count: usize) {
    // Phase 1: fill a colliding table, then carve out the exact tombstone
    // pattern that forces `reserve(1)` down the in-place rehash path.
    let mut map = make_full_map();
    let original_len = map.len();
    assert_eq!(
        (map.len(), map.capacity()),
        (224, 224),
        "this minimized workload is tuned for the validated std/hashbrown layout"
    );

    for i in 1..114 {
        assert_eq!(map.remove(&Key(i)), Some(i));
    }
    assert_eq!(
        map.len(),
        111,
        "setup should leave the expected tombstone pattern"
    );

    // Phase 2: make `BuildHasher::build_hasher` panic during the rehash, then
    // keep using the recovered map.
    PANIC_COUNTER.store(count, Ordering::SeqCst);
    let reserve_panicked = panics_silently(|| {
        map.reserve(1);
    });
    assert!(
        reserve_panicked,
        "the minimized workload should panic during the in-place rehash"
    );

    // Phase 3: a correct table should keep every surviving key reachable and
    // should not start yielding duplicate entries.
    let mut expected_visible_keys: Vec<_> = map.keys().map(|&Key(i)| i).collect();
    let visible_keys: Vec<_> = (0..original_len as u64)
        .filter(|&i| map.get(&Key(i)).copied() == Some(i))
        .collect();
    expected_visible_keys.sort();
    let iter_sample: Vec<_> = map.iter().take(8).map(|(k, v)| (k.0, *v)).collect();
    let distinct_entries = iter_sample.iter().copied().collect::<BTreeSet<_>>();

    assert_eq!(
        map.len(),
        expected_visible_keys.len(),
        "the table length should stay coherent"
    );
    assert_eq!(
        visible_keys, expected_visible_keys,
        "the surviving keys should stay reachable after the caught panic"
    );
    assert_eq!(
        distinct_entries.len(),
        iter_sample.len(),
        "the iterator sample should not contain duplicate entries after the caught panic"
    );
}

#[test]
fn hashmap_reserve_survives_panicking_build_hasher() {
    let _guard = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    if cfg!(miri) {
        for i in [0, 50, 110] {
            hashmap_reserve_survives_panicking_build_hasher_inner(i);
        }
    } else {
        for i in 0..111 {
            hashmap_reserve_survives_panicking_build_hasher_inner(i);
        }
    }
}
