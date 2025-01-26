use super::*;

#[test]
#[cfg(not(miri))]
fn empty() {
    let cache: OnceVec<u32> = OnceVec::default();
    for key in 0..u32::MAX {
        assert!(cache.get(key as usize).is_none());
    }
}

#[test]
fn insert_and_check() {
    let cache: OnceVec<usize> = OnceVec::default();
    for idx in 0..100 {
        cache.insert(idx, idx).unwrap();
    }
    for idx in 0..100 {
        assert_eq!(cache.get(idx), Some(&idx));
    }
}

#[test]
fn sparse_inserts() {
    let cache: OnceVec<u32> = OnceVec::default();
    let end = if cfg!(target_pointer_width = "64") && cfg!(target_os = "linux") {
        // For paged memory, 64-bit systems we should be able to sparsely allocate all of the pages
        // needed for these inserts cheaply (without needing to actually have gigabytes of resident
        // memory).
        31
    } else {
        // Otherwise, still run the test but scaled back:
        //
        // Each slot is <5 bytes, so 2^25 entries (on non-virtual memory systems, like e.g. Windows)
        // will mean 160 megabytes of allocated memory. Going beyond that is probably not reasonable
        // for tests.
        25
    };
    for shift in 0..end {
        let key = 1u32 << shift;
        cache.insert(key as usize, shift).unwrap();
        assert_eq!(cache.get(key as usize), Some(&shift));
    }
}

#[test]
fn concurrent_stress_check() {
    let cache: OnceVec<usize> = OnceVec::default();
    std::thread::scope(|s| {
        for idx in 0..100 {
            let cache = &cache;
            s.spawn(move || {
                cache.insert(idx, idx).unwrap();
            });
        }
    });

    for idx in 0..100 {
        assert_eq!(cache.get(idx), Some(&idx));
    }
}

#[test]
#[cfg(not(miri))]
fn slot_index_exhaustive() {
    let mut prev = None::<(usize, usize, usize)>;
    for idx in 0..=u32::MAX as usize {
        let slot_idx = OnceVec::<()>::to_slab_args(idx);
        if let Some(p) = prev {
            if p.0 == slot_idx.0 {
                assert_eq!(p.2 + 1, slot_idx.2);
            } else {
                assert_eq!(slot_idx.2, 0);
            }
        } else {
            assert_eq!(idx, 0);
            assert_eq!(slot_idx.2, 0);
            assert_eq!(slot_idx.0, 0);
        }

        prev = Some(slot_idx);
    }
}
