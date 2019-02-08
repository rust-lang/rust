//! A singly-linked list.
//!
//! Using this data structure only makes sense under very specific
//! circumstances:
//!
//! - If you have a list that rarely stores more than one element, then this
//!   data-structure can store the element without allocating and only uses as
//!   much space as a `Option<(T, usize)>`. If T can double as the `Option`
//!   discriminant, it will even only be as large as `T, usize`.
//!
//! If you expect to store more than 1 element in the common case, steer clear
//! and use a `Vec<T>`, `Box<[T]>`, or a `SmallVec<T>`.

#[derive(Clone, Hash, Debug, PartialEq)]
pub struct TinyList<T: PartialEq> {
    head: Option<Element<T>>
}

impl<T: PartialEq> TinyList<T> {

    #[inline]
    pub fn new() -> TinyList<T> {
        TinyList {
            head: None
        }
    }

    #[inline]
    pub fn new_single(data: T) -> TinyList<T> {
        TinyList {
            head: Some(Element {
                data,
                next: None,
            })
        }
    }

    #[inline]
    pub fn insert(&mut self, data: T) {
        self.head = Some(Element {
            data,
            next: self.head.take().map(Box::new)
        });
    }

    #[inline]
    pub fn remove(&mut self, data: &T) -> bool {
        self.head = match self.head {
            Some(ref mut head) if head.data == *data => {
                head.next.take().map(|x| *x)
            }
            Some(ref mut head) => return head.remove_next(data),
            None => return false,
        };
        true
    }

    #[inline]
    pub fn contains(&self, data: &T) -> bool {
        if let Some(ref head) = self.head {
            head.contains(data)
        } else {
            false
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        if let Some(ref head) = self.head {
            head.len()
        } else {
            0
        }
    }
}

#[derive(Clone, Hash, Debug, PartialEq)]
struct Element<T: PartialEq> {
    data: T,
    next: Option<Box<Element<T>>>,
}

impl<T: PartialEq> Element<T> {

    fn remove_next(&mut self, data: &T) -> bool {
        let new_next = if let Some(ref mut next) = self.next {
            if next.data != *data {
                return next.remove_next(data)
            } else {
                next.next.take()
            }
        } else {
            return false
        };

        self.next = new_next;

        true
    }

    fn len(&self) -> usize {
        if let Some(ref next) = self.next {
            1 + next.len()
        } else {
            1
        }
    }

    fn contains(&self, data: &T) -> bool {
        if self.data == *data {
            return true
        }

        if let Some(ref next) = self.next {
            next.contains(data)
        } else {
            false
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    extern crate test;
    use test::Bencher;

    #[test]
    fn test_contains_and_insert() {
        fn do_insert(i : u32) -> bool {
            i % 2 == 0
        }

        let mut list = TinyList::new();

        for i in 0 .. 10 {
            for j in 0 .. i {
                if do_insert(j) {
                    assert!(list.contains(&j));
                } else {
                    assert!(!list.contains(&j));
                }
            }

            assert!(!list.contains(&i));

            if do_insert(i) {
                list.insert(i);
                assert!(list.contains(&i));
            }
        }
    }

    #[test]
    fn test_remove_first() {
        let mut list = TinyList::new();
        list.insert(1);
        list.insert(2);
        list.insert(3);
        list.insert(4);
        assert_eq!(list.len(), 4);

        assert!(list.remove(&4));
        assert!(!list.contains(&4));

        assert_eq!(list.len(), 3);
        assert!(list.contains(&1));
        assert!(list.contains(&2));
        assert!(list.contains(&3));
    }

    #[test]
    fn test_remove_last() {
        let mut list = TinyList::new();
        list.insert(1);
        list.insert(2);
        list.insert(3);
        list.insert(4);
        assert_eq!(list.len(), 4);

        assert!(list.remove(&1));
        assert!(!list.contains(&1));

        assert_eq!(list.len(), 3);
        assert!(list.contains(&2));
        assert!(list.contains(&3));
        assert!(list.contains(&4));
    }

    #[test]
    fn test_remove_middle() {
        let mut list = TinyList::new();
        list.insert(1);
        list.insert(2);
        list.insert(3);
        list.insert(4);
        assert_eq!(list.len(), 4);

        assert!(list.remove(&2));
        assert!(!list.contains(&2));

        assert_eq!(list.len(), 3);
        assert!(list.contains(&1));
        assert!(list.contains(&3));
        assert!(list.contains(&4));
    }

    #[test]
    fn test_remove_single() {
        let mut list = TinyList::new();
        list.insert(1);
        assert_eq!(list.len(), 1);

        assert!(list.remove(&1));
        assert!(!list.contains(&1));

        assert_eq!(list.len(), 0);
    }

    #[bench]
    fn bench_insert_empty(b: &mut Bencher) {
        b.iter(|| {
            let mut list = TinyList::new();
            list.insert(1);
        })
    }

    #[bench]
    fn bench_insert_one(b: &mut Bencher) {
        b.iter(|| {
            let mut list = TinyList::new_single(0);
            list.insert(1);
        })
    }

    #[bench]
    fn bench_remove_empty(b: &mut Bencher) {
        b.iter(|| {
            TinyList::new().remove(&1)
        });
    }

    #[bench]
    fn bench_remove_unknown(b: &mut Bencher) {
        b.iter(|| {
            TinyList::new_single(0).remove(&1)
        });
    }

    #[bench]
    fn bench_remove_one(b: &mut Bencher) {
        b.iter(|| {
            TinyList::new_single(1).remove(&1)
        });
    }
}
