use std::collections::LinkedList;

#[test]
fn test_hash() {
    use crate::hash;

    let mut x = LinkedList::new();
    let mut y = LinkedList::new();

    assert!(hash(&x) == hash(&y));

    x.push_back(1);
    x.push_back(2);
    x.push_back(3);

    y.push_front(3);
    y.push_front(2);
    y.push_front(1);

    assert!(hash(&x) == hash(&y));
}
