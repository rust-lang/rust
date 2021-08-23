#![cfg(not(miri))] // FIXME: takes too long

use hashbrown::HashSet;
use rand::{distributions::Alphanumeric, rngs::SmallRng, Rng, SeedableRng};
use std::iter;

#[test]
fn test_hashset_insert_remove() {
    let mut m: HashSet<Vec<char>> = HashSet::new();
    let seed: [u8; 32] = [
        130, 220, 246, 217, 111, 124, 221, 189, 190, 234, 121, 93, 67, 95, 100, 43, // again
        130, 220, 246, 217, 111, 124, 221, 189, 190, 234, 121, 93, 67, 95, 100, 43,
    ];

    let rng = &mut SmallRng::from_seed(seed);
    let tx: Vec<Vec<char>> = iter::repeat_with(|| {
        rng.sample_iter(&Alphanumeric)
            .take(32)
            .map(char::from)
            .collect()
    })
    .take(4096)
    .collect();

    // more readable with explicit `true` / `false`
    #[allow(clippy::bool_assert_comparison)]
    for _ in 0..32 {
        for x in &tx {
            assert_eq!(m.contains(x), false);
            assert_eq!(m.insert(x.clone()), true);
        }
        for (i, x) in tx.iter().enumerate() {
            println!("removing {} {:?}", i, x);
            assert_eq!(m.remove(x), true);
        }
    }
}
