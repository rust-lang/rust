use rustc_data_structures::unord::UnordSet;
use rustc_index::bit_set::BitMatrix;
use rustc_middle::mono::CastRelevantLifetimes;

use crate::cast_sensitivity::CallerOutlivesEnv;

// ── CallerOutlivesEnv tests ─────────────────────────────────────

/// Build an outlives reachability `BitMatrix` from raw pairs using the
/// same Floyd-Warshall algorithm as `outlives_reachability`, then call
/// `f` with a borrowed `CallerOutlivesEnv`. Avoids leaking memory and
/// keeps the construction logic in test code.
fn with_env(pairs: &[(usize, usize)], f: impl FnOnce(&CallerOutlivesEnv<'_>)) {
    let max_idx =
        pairs.iter().flat_map(|&(l, s)| [l, s]).filter(|&v| v != usize::MAX).max().unwrap_or(0);
    let dim = if pairs.is_empty() { 1 } else { max_idx + 2 };
    let static_idx = dim - 1;

    let mut reach = BitMatrix::new(dim, dim);
    for i in 0..dim {
        reach.insert(i, i);
    }
    for j in 0..dim {
        reach.insert(static_idx, j);
    }
    let remap = |idx: usize| if idx == usize::MAX { static_idx } else { idx };
    for &(l, s) in pairs {
        reach.insert(remap(l), remap(s));
    }
    for k in 0..dim {
        for i in 0..dim {
            if reach.contains(i, k) {
                reach.union_rows(k, i);
            }
        }
    }

    let env = CallerOutlivesEnv::from_raw(&reach, dim);
    f(&env);
}

#[test]
fn outlives_reflexive() {
    with_env(&[], |env| {
        assert!(env.outlives(0, 0));
        assert!(env.outlives(usize::MAX, usize::MAX));
    });
}

#[test]
fn outlives_direct() {
    // 0 : 1 (0 outlives 1)
    with_env(&[(0, 1)], |env| {
        assert!(env.outlives(0, 1));
        assert!(!env.outlives(1, 0)); // not symmetric
    });
}

#[test]
fn outlives_transitive() {
    // 0 : 1, 1 : 2 → 0 : 2 transitively
    with_env(&[(0, 1), (1, 2)], |env| {
        assert!(env.outlives(0, 1));
        assert!(env.outlives(1, 2));
        assert!(env.outlives(0, 2)); // transitive
        assert!(!env.outlives(2, 0));
        assert!(!env.outlives(2, 1));
    });
}

#[test]
fn outlives_cycle() {
    // 0 : 1, 1 : 0 → mutual outlives (same equivalence class)
    with_env(&[(0, 1), (1, 0)], |env| {
        assert!(env.outlives(0, 1));
        assert!(env.outlives(1, 0));
    });
}

#[test]
fn outlives_disconnected() {
    // 0 : 1, 2 : 3 → no relationship between {0,1} and {2,3}
    with_env(&[(0, 1), (2, 3)], |env| {
        assert!(env.outlives(0, 1));
        assert!(env.outlives(2, 3));
        assert!(!env.outlives(0, 2));
        assert!(!env.outlives(0, 3));
        assert!(!env.outlives(1, 2));
        assert!(!env.outlives(3, 0));
    });
}

#[test]
fn outlives_diamond() {
    // 0 : 1, 0 : 2, 1 : 3, 2 : 3 → 0 : 3 through either path
    with_env(&[(0, 1), (0, 2), (1, 3), (2, 3)], |env| {
        assert!(env.outlives(0, 3)); // through 0→1→3 or 0→2→3
        assert!(!env.outlives(3, 0));
        assert!(!env.outlives(1, 2)); // no direct or transitive path 1→2
        assert!(!env.outlives(2, 1)); // no direct or transitive path 2→1
    });
}

#[test]
fn outlives_static() {
    // 5 : usize::MAX (param 5 outlives 'static)
    with_env(&[(5, usize::MAX)], |env| {
        assert!(env.outlives(5, usize::MAX));
        // 'static outlives everything by construction in the reachability matrix.
        assert!(env.outlives(usize::MAX, 5));
    });
}

#[test]
fn outlives_long_chain() {
    // 0 : 1 : 2 : 3 : 4 → 0 : 4 transitively
    with_env(&[(0, 1), (1, 2), (2, 3), (3, 4)], |env| {
        assert!(env.outlives(0, 4));
        assert!(env.outlives(0, 3));
        assert!(env.outlives(1, 4));
        assert!(!env.outlives(4, 0));
    });
}

// ── CastRelevantLifetimes::max_walk_order_position tests ────────

#[test]
fn max_walk_order_empty() {
    let crl = CastRelevantLifetimes { mappings: UnordSet::new() };
    assert_eq!(crl.max_walk_order_position(), 0);
}

// Note: non-empty max_walk_order_position tests require TyCtxt for
// LifetimeBVToParamMapping interning and are covered by integration tests.

// ── Transitive reduction tests ──────────────────────────────────
// Test the transitive reduction logic extracted from augment_callee.

/// Apply the same transitive reduction algorithm used in augment_callee.
fn transitive_reduction(pairs: &[(usize, usize)]) -> Vec<(usize, usize)> {
    let mut minimal: Vec<(usize, usize)> = Vec::new();
    for &(l, s) in pairs {
        let is_redundant = pairs.iter().any(|&(l2, mid)| {
            l2 == l && mid != s && pairs.iter().any(|&(l3, s3)| l3 == mid && s3 == s)
        });
        if !is_redundant {
            minimal.push((l, s));
        }
    }
    minimal.sort();
    minimal.dedup();
    minimal
}

#[test]
fn transitive_reduction_empty() {
    assert_eq!(transitive_reduction(&[]), vec![]);
}

#[test]
fn transitive_reduction_single() {
    assert_eq!(transitive_reduction(&[(0, 1)]), vec![(0, 1)]);
}

#[test]
fn transitive_reduction_chain() {
    // 0 : 1, 1 : 2, 0 : 2 → remove 0 : 2 (redundant via 0→1→2)
    let result = transitive_reduction(&[(0, 1), (1, 2), (0, 2)]);
    assert_eq!(result, vec![(0, 1), (1, 2)]);
}

#[test]
fn transitive_reduction_no_redundancy() {
    // 0 : 1, 2 : 3 → both kept (no transitive path)
    let result = transitive_reduction(&[(0, 1), (2, 3)]);
    assert_eq!(result, vec![(0, 1), (2, 3)]);
}

#[test]
fn transitive_reduction_diamond() {
    // 0 : 1, 0 : 2, 1 : 3, 2 : 3 → all kept (no single intermediate)
    // 0→3 doesn't exist, so no redundancy to remove
    let result = transitive_reduction(&[(0, 1), (0, 2), (1, 3), (2, 3)]);
    assert_eq!(result, vec![(0, 1), (0, 2), (1, 3), (2, 3)]);
}

#[test]
fn transitive_reduction_diamond_with_shortcut() {
    // 0 : 1, 0 : 2, 1 : 3, 2 : 3, 0 : 3 → remove 0 : 3
    let result = transitive_reduction(&[(0, 1), (0, 2), (1, 3), (2, 3), (0, 3)]);
    assert_eq!(result, vec![(0, 1), (0, 2), (1, 3), (2, 3)]);
}

#[test]
fn transitive_reduction_longer_chain() {
    // 0 : 1, 1 : 2, 2 : 3, 0 : 2, 0 : 3, 1 : 3
    // → remove 0:2 (via 0→1→2), 0:3 (via 0→1→3), 1:3 (via 1→2→3)
    let result = transitive_reduction(&[(0, 1), (1, 2), (2, 3), (0, 2), (0, 3), (1, 3)]);
    assert_eq!(result, vec![(0, 1), (1, 2), (2, 3)]);
}

#[test]
fn transitive_reduction_with_static() {
    // 0 : 1, 0 : MAX, 1 : MAX → remove 0 : MAX (via 0→1→MAX)
    let s = usize::MAX;
    let result = transitive_reduction(&[(0, 1), (0, s), (1, s)]);
    assert_eq!(result, vec![(0, 1), (1, s)]);
}

#[test]
fn transitive_reduction_duplicates() {
    // Duplicate entries should be deduplicated.
    let result = transitive_reduction(&[(0, 1), (0, 1), (1, 2)]);
    assert_eq!(result, vec![(0, 1), (1, 2)]);
}
