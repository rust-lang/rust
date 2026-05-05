use std::sync::atomic::Ordering;

use super::*;

#[test]
#[cfg(target_pointer_width = "64")]
fn linear_vec_cache_insert_and_check() {
    let cache: LinearVecCache<u32, u32, u32> = LinearVecCache::default();
    cache.complete(0, 1, 2);
    assert_eq!(cache.lookup(&0), Some((1, 2)));
}

#[test]
#[cfg(target_pointer_width = "64")]
fn linear_vec_cache_sparse_inserts() {
    let cache: LinearVecCache<u32, u8, u32> = LinearVecCache::default();
    for shift in 0..31 {
        let key = 1u32 << shift;
        cache.complete(key, shift, key);
        assert_eq!(cache.lookup(&key), Some((shift, key)));
    }
}

#[test]
#[cfg(target_pointer_width = "64")]
fn linear_vec_cache_uncommitted_miss() {
    let cache: LinearVecCache<u32, u32, u32> = LinearVecCache::default();
    assert_eq!(cache.lookup(&u32::MAX), None);
}

#[test]
#[cfg(target_pointer_width = "64")]
fn linear_vec_push_and_get() {
    let vec = LinearVec::<u32>::new();
    assert_eq!(vec.len(), 0);

    let idx0 = vec.push(11u32);
    let idx1 = vec.push(22u32);

    assert_eq!(idx0, 0);
    assert_eq!(idx1, 1);
    assert_eq!(vec.len(), 2);
    assert_eq!(vec.get(0), Some(&11));
    assert_eq!(vec.get(1), Some(&22));
    assert_eq!(vec.get(2), None);
}

#[test]
#[cfg(target_pointer_width = "64")]
fn linear_storage_commit_index() {
    let storage = LinearStorage::<u64>::new();
    let _ = storage.ensure_committed_for_index(0);
    assert_eq!(unsafe { storage.get_copy(0) }, None);

    assert!(storage.put(10, 77, 5));
    assert_eq!(unsafe { storage.get_copy(10) }, Some((77, 5)));
}

#[test]
#[cfg(target_pointer_width = "64")]
fn linear_storage_repeated_commit_request_is_noop() {
    let storage = LinearStorage::<u64>::new();

    assert!(storage.ensure_committed_for_index(10));
    assert!(!storage.ensure_committed_for_index(10));
}

#[test]
#[cfg(target_pointer_width = "64")]
fn linear_vec_cache_for_each_tolerates_in_flight_present_entry() {
    let cache: LinearVecCache<u32, u32, u32> = LinearVecCache::default();

    cache.present.len.store(1, Ordering::Release);

    let mut visited = 0;
    cache.for_each(&mut |_, _, _| visited += 1);
    assert_eq!(visited, 0);
}
