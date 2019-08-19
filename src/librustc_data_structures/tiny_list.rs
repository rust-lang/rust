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

#[cfg(test)]
mod tests;

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
