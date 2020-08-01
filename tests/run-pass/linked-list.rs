#![feature(linked_list_cursors)]
use std::collections::LinkedList;

fn list_from<T: Clone>(v: &[T]) -> LinkedList<T> {
    v.iter().cloned().collect()
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
    assert_eq!(m.into_iter().collect::<Vec<_>>(),
               [-10, -2, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 99]);
}
