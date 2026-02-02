use super::*;

#[test]
#[cfg(not(miri))]
fn vec_cache_empty() {
    let cache: VecCache<u32, u32, u32> = VecCache::default();
    for key in 0..u32::MAX {
        assert!(cache.lookup(&key).is_none());
    }
}

#[test]
fn vec_cache_insert_and_check() {
    let cache: VecCache<u32, u32, u32> = VecCache::default();
    cache.complete(0, 1, 2);
    assert_eq!(cache.lookup(&0), Some((1, 2)));
}

#[test]
fn sparse_inserts() {
    let cache: VecCache<u32, u8, u32> = VecCache::default();
    let end = if cfg!(target_pointer_width = "64") && cfg!(target_os = "linux") {
        // For paged memory, 64-bit systems we should be able to sparsely allocate all of the pages
        // needed for these inserts cheaply (without needing to actually have gigabytes of resident
        // memory).
        31
    } else {
        // Otherwise, still run the test but scaled back:
        //
        // Each slot is 5 bytes, so 2^25 entries (on non-virtual memory systems, like e.g. Windows) will
        // mean 160 megabytes of allocated memory. Going beyond that is probably not reasonable for
        // tests.
        25
    };
    for shift in 0..end {
        let key = 1u32 << shift;
        cache.complete(key, shift, key);
        assert_eq!(cache.lookup(&key), Some((shift, key)));
    }
}

#[test]
fn concurrent_stress_check() {
    let cache: VecCache<u32, u32, u32> = VecCache::default();
    std::thread::scope(|s| {
        for idx in 0..100 {
            let cache = &cache;
            s.spawn(move || {
                cache.complete(idx, idx, idx);
            });
        }
    });

    for idx in 0..100 {
        assert_eq!(cache.lookup(&idx), Some((idx, idx)));
    }
}

#[test]
fn slot_entries_table() {
    assert_eq!(
        ENTRIES_BY_BUCKET,
        [
            4096, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576, 2097152,
            4194304, 8388608, 16777216, 33554432, 67108864, 134217728, 268435456, 536870912,
            1073741824, 2147483648
        ]
    );
}

#[test]
#[cfg(not(miri))]
fn slot_index_exhaustive() {
    let mut buckets = [0u32; 21];
    for idx in 0..=u32::MAX {
        buckets[SlotIndex::from_index(idx).bucket_idx] += 1;
    }
    let slot_idx = SlotIndex::from_index(0);
    assert_eq!(slot_idx.index_in_bucket, 0);
    assert_eq!(slot_idx.bucket_idx, 0);
    let mut prev = slot_idx;
    for idx in 1..=u32::MAX {
        let slot_idx = SlotIndex::from_index(idx);
        if prev.bucket_idx == slot_idx.bucket_idx {
            assert_eq!(prev.index_in_bucket + 1, slot_idx.index_in_bucket);
        } else {
            assert_eq!(slot_idx.index_in_bucket, 0);
        }

        assert_eq!(buckets[slot_idx.bucket_idx], slot_idx.entries as u32);
        assert_eq!(ENTRIES_BY_BUCKET[slot_idx.bucket_idx], slot_idx.entries, "{}", idx);

        prev = slot_idx;
    }
}
