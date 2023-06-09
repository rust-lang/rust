//@revisions: stack tree
//@[tree]compile-flags: -Zmiri-tree-borrows
#![feature(linked_list_cursors)]
use std::collections::LinkedList;

fn list_from<T: Clone>(v: &[T]) -> LinkedList<T> {
    v.iter().cloned().collect()
}

// Gather all references from a mutable iterator and make sure Miri notices if
// using them is dangerous.
fn test_all_refs<'a, T: 'a>(dummy: &mut T, iter: impl Iterator<Item = &'a mut T>) {
    // Gather all those references.
    let mut refs: Vec<&mut T> = iter.collect();
    // Use them all. Twice, to be sure we got all interleavings.
    for r in refs.iter_mut() {
        std::mem::swap(dummy, r);
    }
    for r in refs {
        std::mem::swap(dummy, r);
    }
}

fn main() {
    let mut m = list_from(&[0, 2, 4, 6, 8]);
    let len = m.len();
    {
        let mut it = m.cursor_front_mut();
        it.insert_before(-2);
        loop {
            match it.current().copied() {
                None => break,
                Some(elt) => {
                    match it.peek_next() {
                        Some(x) => assert_eq!(*x, elt + 2),
                        None => assert_eq!(8, elt),
                    }
                    it.insert_after(elt + 1);
                    it.move_next(); // Move by 2 to skip the one we inserted.
                    it.move_next();
                }
            }
        }
        it.insert_before(99);
        it.insert_after(-10);
    }

    assert_eq!(m.len(), 3 + len * 2);
    let mut m2 = m.clone();
    assert_eq!(m.into_iter().collect::<Vec<_>>(), [-10, -2, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 99]);

    test_all_refs(&mut 13, m2.iter_mut());
}
