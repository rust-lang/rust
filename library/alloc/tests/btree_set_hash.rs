use std::collections::BTreeSet;

#[test]
fn test_hash() {
    use crate::hash;

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
