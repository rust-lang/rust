use std::collections::BTreeSet;

use crate::hash;

#[test]
fn test_hash() {
    let mut x = BTreeSet::new();
    let mut y = BTreeSet::new();

    x.insert(1);
    x.insert(2);
    x.insert(3);

    y.insert(3);
    y.insert(2);
    y.insert(1);

    assert_eq!(hash(&x), hash(&y));
}

#[test]
fn test_prefix_free() {
    let x = BTreeSet::from([1, 2, 3]);
    let y = BTreeSet::<i32>::new();

    // If hashed by iteration alone, `(x, y)` and `(y, x)` would visit the same
    // order of elements, resulting in the same hash. But now that we also hash
    // the length, they get distinct sequences of hashed data.
    assert_ne!(hash(&(&x, &y)), hash(&(&y, &x)));
}
