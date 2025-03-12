#![cfg(test)]

use crate::collect::release;
use crate::sync_push_vec::SyncPushVec;

#[test]
fn test_iter() {
    let mut m = SyncPushVec::new();
    m.write().push(1);
    m.write().push(2);
    assert_eq!(
        m.write()
            .read()
            .as_slice()
            .iter()
            .copied()
            .collect::<Vec<i32>>(),
        vec![1, 2]
    );
}

#[test]
fn test_high_align() {
    #[repr(align(128))]
    #[derive(Clone)]
    struct A(u8);
    let mut m = SyncPushVec::<A>::new();
    for _a in m.write().read().as_slice() {}
    m.write().push(A(1));
    for _a in m.write().read().as_slice() {}
}

#[test]
fn test_low_align() {
    let mut m = SyncPushVec::<u8>::with_capacity(1);
    m.write().push(1);
}

#[test]
fn test_insert() {
    let m = SyncPushVec::new();
    assert_eq!(m.lock().read().len(), 0);
    m.lock().push(2);
    assert_eq!(m.lock().read().len(), 1);
    m.lock().push(5);
    assert_eq!(m.lock().read().len(), 2);
    assert_eq!(m.lock().read().as_slice()[0], 2);
    assert_eq!(m.lock().read().as_slice()[1], 5);

    release();
}

#[test]
fn test_replace() {
    let m = SyncPushVec::new();
    m.lock().push(2);
    m.lock().push(5);
    assert_eq!(m.lock().read().as_slice(), [2, 5]);
    m.lock().replace(vec![3].into_iter(), 0);
    assert_eq!(m.lock().read().as_slice(), [3]);
    m.lock().replace(vec![].into_iter(), 0);
    assert_eq!(m.lock().read().as_slice(), []);
    release();
}

#[test]
fn test_expand() {
    let m = SyncPushVec::new();

    assert_eq!(m.lock().read().len(), 0);

    let mut i = 0;
    let old_raw_cap = m.lock().read().capacity();
    while old_raw_cap == m.lock().read().capacity() {
        m.lock().push(i);
        i += 1;
    }

    assert_eq!(m.lock().read().len(), i);

    release();
}
