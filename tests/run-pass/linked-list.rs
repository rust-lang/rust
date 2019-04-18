#![feature(linked_list_extras)]
use std::collections::LinkedList;

fn list_from<T: Clone>(v: &[T]) -> LinkedList<T> {
    v.iter().cloned().collect()
}
    
fn main() {
    let mut m = list_from(&[0, 2, 4, 6, 8]);
    let len = m.len();
    {
        let mut it = m.iter_mut();
        it.insert_next(-2);
        loop {
            match it.next() {
                None => break,
                Some(elt) => {
                    it.insert_next(*elt + 1);
                    match it.peek_next() {
                        Some(x) => assert_eq!(*x, *elt + 2),
                        None => assert_eq!(8, *elt),
                    }
                }
            }
        }
        it.insert_next(0);
        it.insert_next(1);
    }

    assert_eq!(m.len(), 3 + len * 2);
    assert_eq!(m.into_iter().collect::<Vec<_>>(),
               [-2, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1]);
}
