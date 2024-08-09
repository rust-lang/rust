use super::*;

#[test]
fn vec_cache_empty() {
    let cache: VecCache<u32, u32, u32> = VecCache::default();
    for key in 0..=u32::MAX {
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
fn slot_index_exhaustive() {
    let mut buckets = [0u32; 21];
    for idx in 0..=u32::MAX {
        buckets[SlotIndex::from_index(idx).bucket_idx] += 1;
    }
    let mut prev = None::<SlotIndex>;
    for idx in 0..u32::MAX {
        let slot_idx = SlotIndex::from_index(idx);
        if let Some(p) = prev {
            if p.bucket_idx == slot_idx.bucket_idx {
                assert_eq!(p.index_in_bucket + 1, slot_idx.index_in_bucket);
            } else {
                assert_eq!(slot_idx.index_in_bucket, 0);
            }
        } else {
            assert_eq!(idx, 0);
            assert_eq!(slot_idx.index_in_bucket, 0);
            assert_eq!(slot_idx.bucket_idx, 0);
        }

        assert_eq!(buckets[slot_idx.bucket_idx], slot_idx.entries as u32);

        prev = Some(slot_idx);
    }
}
