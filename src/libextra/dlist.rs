
//! A doubly-linked list with owned nodes.
//!
//! The List allows pushing and popping elements at either end.


// List is constructed like a singly-linked list over the field `next`.
// including the last link being None; each Node owns its `next` field.
//
// Backlinks over List::prev are raw pointers that form a full chain in
// the reverse direction.


use std::cast;
use std::cmp;
use std::util;
use std::iterator::FromIterator;

/// A doubly-linked list
pub struct List<T> {
    priv length: uint,
    priv list_head: Link<T>,
    priv list_tail: Rawlink<T>,
}

type Link<T> = Option<~Node<T>>;
type Rawlink<T> = Option<&'static Node<T>>;
// Rawlink uses &'static to have a small Option<&'> represenation.
// FIXME: Use a raw pointer like *mut Node if possible.
// FIXME: Causes infinite recursion in %? repr

struct Node<T> {
    priv next: Link<T>,
    priv prev: Rawlink<T>,
    priv value: T,
}

/// List iterator
pub struct ForwardIterator<'self, T> {
    priv list: &'self List<T>,
    priv next: &'self Link<T>,
}

/// List reverse iterator
pub struct ReverseIterator<'self, T> {
    priv list: &'self List<T>,
    priv next: Rawlink<T>,
}

/// List mutable iterator
pub struct MutForwardIterator<'self, T> {
    priv list: &'self mut List<T>,
    priv curs: Rawlink<T>,
}

/// List mutable reverse iterator
pub struct MutReverseIterator<'self, T> {
    priv list: &'self mut List<T>,
    priv next: Rawlink<T>,
}

/// List consuming iterator
pub struct ConsumeIterator<T> {
    priv list: List<T>
}

/// List reverse consuming iterator
pub struct ConsumeRevIterator<T> {
    priv list: List<T>
}

impl<T> Container for List<T> {
    /// O(1)
    fn is_empty(&self) -> bool {
        self.list_head.is_none()
    }
    /// O(1)
    fn len(&self) -> uint {
        self.length
    }
}

impl<T> Mutable for List<T> {
    /// Remove all elements from the List
    ///
    /// O(N)
    fn clear(&mut self) {
        *self = List::new()
    }
}

/// Cast the raw link into a borrowed ref
fn resolve_rawlink<T>(lnk: &'static Node<T>) -> &mut Node<T> {
    unsafe { cast::transmute_mut(lnk) }
}
fn rawlink<T>(n: &mut Node<T>) -> Rawlink<T> {
    Some(unsafe { cast::transmute(n) })
}

impl<T> List<T> {
    /// Create an empty List
    #[inline]
    pub fn new() -> List<T> {
        List{list_head: None, list_tail: None, length: 0}
    }

    /// Provide a reference to the front element, or None if the list is empty
    pub fn peek_front<'a>(&'a self) -> Option<&'a T> {
        self.list_head.chain_ref(|x| Some(&x.value))
    }

    /// Provide a mutable reference to the front element, or None if the list is empty
    pub fn peek_front_mut<'a>(&'a mut self) -> Option<&'a mut T> {
        match self.list_head {
            None => None,
            Some(ref mut head) => Some(&mut head.value),
        }
    }

    /// Provide a reference to the back element, or None if the list is empty
    pub fn peek_back<'a>(&'a self) -> Option<&'a T> {
        match self.list_tail {
            None => None,
            Some(tail) => Some(&resolve_rawlink(tail).value),
        }
    }

    /// Provide a mutable reference to the back element, or None if the list is empty
    pub fn peek_back_mut<'a>(&'a mut self) -> Option<&'a mut T> {
        match self.list_tail {
            None => None,
            Some(tail) => Some(&mut resolve_rawlink(tail).value),
        }
    }

    /// Add an element last in the list
    ///
    /// O(1)
    pub fn push_back(&mut self, elt: T) {
        match self.list_tail {
            None => return self.push_front(elt),
            Some(rtail) => {
                let mut new_tail = ~Node{value: elt, next: None, prev: self.list_tail};
                self.list_tail = rawlink(new_tail);
                let tail = resolve_rawlink(rtail);
                tail.next = Some(new_tail);
            }
        }
        self.length += 1;
    }

    /// Remove the last element and return it, or None if the list is empty
    ///
    /// O(1)
    #[inline]
    pub fn pop_back(&mut self) -> Option<T> {
        match self.list_tail {
            None => None,
            Some(rtail) => {
                self.length -= 1;
                let tail = resolve_rawlink(rtail);
                let tail_own = match tail.prev {
                    None => {
                        self.list_tail = None;
                        self.list_head.swap_unwrap()
                    },
                    Some(rtail_prev) => {
                        self.list_tail = tail.prev;
                        resolve_rawlink(rtail_prev).next.swap_unwrap()
                    }
                };
                Some(tail_own.value)
            }
        }
    }

    /// Add an element first in the list
    ///
    /// O(1)
    pub fn push_front(&mut self, elt: T) {
        let mut new_head = ~Node{value: elt, next: None, prev: None};
        match self.list_head {
            None => {
                self.list_tail = rawlink(new_head);
                self.list_head = Some(new_head);
            }
            Some(ref mut head) => {
                head.prev = rawlink(new_head);
                util::swap(head, &mut new_head);
                head.next = Some(new_head);
            }
        }
        self.length += 1;
    }

    /// Remove the first element and return it, or None if the list is empty
    ///
    /// O(1)
    pub fn pop_front(&mut self) -> Option<T> {
        match self.list_head {
            None => None,
            ref mut head @ Some(*) => {
                self.length -= 1;
                match *head.swap_unwrap() {
                    Node{value: value, next: Some(next), prev: _} => {
                        let mut mnext = next;
                        mnext.prev = None;
                        *head = Some(mnext);
                        Some(value)
                    }
                    Node{value: value, next: None, prev: _} => {
                        self.list_tail = None;
                        *head = None;
                        Some(value)
                    }
                }
            }
        }
    }

    /// Add all elements from `other` to the end of the list
    ///
    /// O(1)
    pub fn append(&mut self, other: List<T>) {
        match self.list_tail {
            None => *self = other,
            Some(rtail) => {
                match other {
                    List{list_head: None, list_tail: _, length: _} => return,
                    List{list_head: Some(node), list_tail: o_tail, length: o_length} => {
                        let mut lnk_node = node;
                        let tail = resolve_rawlink(rtail);
                        lnk_node.prev = self.list_tail;
                        tail.next = Some(lnk_node);
                        self.list_tail = o_tail;
                        self.length += o_length;
                    }
                }
            }
        }
    }

    /// Add all elements from `other` to the beginning of the list
    ///
    /// O(1)
    pub fn prepend(&mut self, mut other: List<T>) {
        util::swap(self, &mut other);
        self.append(other);
    }

    /// Insert `elt` before the first `x` in the list where `f(x, elt)` is true,
    /// or at the end.
    ///
    /// O(N)
    #[inline]
    pub fn insert_before(&mut self, elt: T, f: &fn(&T, &T) -> bool) {
        {
            let mut it = self.mut_iter();
            loop {
                match it.next() {
                    None => break,
                    Some(x) => if f(x, &elt) { it.insert_before(elt); return }
                }
            }
        }
        self.push_back(elt);
    }

    /// Merge, using the function `f`; take `a` if `f(a, b)` is true, else `b`.
    ///
    /// O(max(N, M))
    pub fn merge(&mut self, mut other: List<T>, f: &fn(&T, &T) -> bool) {
        {
            let mut it = self.mut_iter();
            loop {
                match (it.next(), other.peek_front()) {
                    (None   , _      ) => break,
                    (_      , None   ) => return,
                    (Some(x), Some(y)) => if f(x, y) { loop }
                }
                it.insert_before(other.pop_front().unwrap());
            }
        }
        self.append(other);
    }


    /// Provide a forward iterator
    pub fn iter<'a>(&'a self) -> ForwardIterator<'a, T> {
        ForwardIterator{list: self, next: &self.list_head}
    }

    /// Provide a reverse iterator
    pub fn rev_iter<'a>(&'a self) -> ReverseIterator<'a, T> {
        ReverseIterator{list: self, next: self.list_tail}
    }

    /// Provide a forward iterator with mutable references
    pub fn mut_iter<'a>(&'a mut self) -> MutForwardIterator<'a, T> {
        MutForwardIterator{list: self, curs: None}
    }

    /// Provide a reverse iterator with mutable references
    pub fn mut_rev_iter<'a>(&'a mut self) -> MutReverseIterator<'a, T> {
        MutReverseIterator{list: self, next: self.list_tail}
    }


    /// Consume the list into an iterator yielding elements by value
    pub fn consume_iter(self) -> ConsumeIterator<T> {
        ConsumeIterator{list: self}
    }

    /// Consume the list into an iterator yielding elements by value, in reverse
    pub fn consume_rev_iter(self) -> ConsumeRevIterator<T> {
        ConsumeRevIterator{list: self}
    }
}

/// Insert sorted in ascending order
///
/// O(N)
impl<T: cmp::TotalOrd> List<T> {
    fn insert_ordered(&mut self, elt: T) {
        self.insert_before(elt, |a, b| a.cmp(b) != cmp::Less);
    }
}

impl<'self, A> Iterator<&'self A> for ForwardIterator<'self, A> {
    #[inline]
    fn next(&mut self) -> Option<&'self A> {
        match *self.next {
            None => None,
            Some(ref next) => {
                self.next = &next.next;
                Some(&next.value)
            }
        }
    }

    fn size_hint(&self) -> (uint, Option<uint>) {
        (0, Some(self.list.length))
    }
}

// MutForwardIterator is different because it implements ListInsertCursor,
// and can modify the list during traversal, used in insert_when and merge.
impl<'self, A> Iterator<&'self mut A> for MutForwardIterator<'self, A> {
    #[inline]
    fn next(&mut self) -> Option<&'self mut A> {
        match self.curs {
            None => {
                match self.list.list_head {
                    None => None,
                    Some(ref mut head) => {
                        self.curs = rawlink(&mut **head);
                        Some(&mut head.value)
                    }
                }
            }
            Some(rcurs) => {
                match resolve_rawlink(rcurs).next {
                    None => None,
                    Some(ref mut head) => {
                        self.curs = rawlink(&mut **head);
                        Some(&mut head.value)
                    }
                }
            }
        }
    }

    fn size_hint(&self) -> (uint, Option<uint>) {
        (0, Some(self.list.length))
    }
}

impl<'self, A> Iterator<&'self A> for ReverseIterator<'self, A> {
    #[inline]
    fn next(&mut self) -> Option<&'self A> {
        match self.next {
            None => None,
            Some(rnext) => {
                let prev = resolve_rawlink(rnext);
                self.next = prev.prev;
                Some(&prev.value)
            }
        }
    }

    fn size_hint(&self) -> (uint, Option<uint>) {
        (0, Some(self.list.length))
    }
}

impl<'self, A> Iterator<&'self mut A> for MutReverseIterator<'self, A> {
    #[inline]
    fn next(&mut self) -> Option<&'self mut A> {
        match self.next {
            None => None,
            Some(rnext) => {
                let prev = resolve_rawlink(rnext);
                self.next = prev.prev;
                Some(&mut prev.value)
            }
        }
    }

    fn size_hint(&self) -> (uint, Option<uint>) {
        (0, Some(self.list.length))
    }
}

// XXX: Should this be `pub`?
trait ListInsertCursor<A> {
    /// Insert `elt` just previous to the most recently yielded element
    fn insert_before(&mut self, elt: A);
}

impl<'self, A> ListInsertCursor<A> for MutForwardIterator<'self, A> {
    fn insert_before(&mut self, elt: A) {
        match self.curs {
            None => self.list.push_front(elt),
            Some(rcurs) => {
                let node = resolve_rawlink(rcurs);
                let prev_node = match node.prev {
                    None => return self.list.push_front(elt),  // at head
                    Some(rprev) => resolve_rawlink(rprev),
                };
                let mut node_own = prev_node.next.swap_unwrap();
                let mut ins_node = ~Node{value: elt,
                                         next: None,
                                         prev: rawlink(prev_node)};
                node_own.prev = rawlink(ins_node);
                ins_node.next = Some(node_own);
                prev_node.next = Some(ins_node);
                self.list.length += 1;
            }
        }
    }
}

impl<A> Iterator<A> for ConsumeIterator<A> {
    fn next(&mut self) -> Option<A> { self.list.pop_front() }
    fn size_hint(&self) -> (uint, Option<uint>) {
        (self.list.length, Some(self.list.length))
    }
}

impl<A> Iterator<A> for ConsumeRevIterator<A> {
    fn next(&mut self) -> Option<A> { self.list.pop_back() }
    fn size_hint(&self) -> (uint, Option<uint>) {
        (self.list.length, Some(self.list.length))
    }
}

impl<A, T: Iterator<A>> FromIterator<A, T> for List<A> {
    fn from_iterator(iterator: &mut T) -> List<A> {
        let mut ret = List::new();
        for iterator.advance |elt| { ret.push_back(elt); }
        ret
    }
}

impl<A: Eq> Eq for List<A> {
    fn eq(&self, other: &List<A>) -> bool {
        self.len() == other.len() &&
            self.iter().zip(other.iter()).all(|(a, b)| a.eq(b))
    }
    fn ne(&self, other: &List<A>) -> bool {
        !self.eq(other)
    }
}

impl<A: Clone> Clone for List<A> {
    fn clone(&self) -> List<A> {
        self.iter().transform(|x| x.clone()).collect()
    }
}

#[cfg(test)]
fn check_links<T>(list: &List<T>) {
    let mut len = 0u;
    let mut last_ptr: Option<&Node<T>> = None;
    let mut node_ptr: &Node<T>;
    match list.list_head {
        None => { assert_eq!(0u, list.length); return }
        Some(ref node) => node_ptr = &**node,
    }
    loop {
        match (last_ptr, node_ptr.prev) {
            (None   , None      ) => {}
            (None   , _         ) => fail!("prev link for list_head"),
            (Some(p), Some(pptr)) => {
                assert_eq!((p as *Node<T>) as uint, pptr as *Node<T> as uint);
            }
            _ => fail!("prev link is none, not good"),
        }
        match node_ptr.next {
            Some(ref next) => {
                last_ptr = Some(node_ptr);
                node_ptr = &**next;
                len += 1;
            }
            None => {
                len += 1;
                break;
            }
        }
    }
    assert_eq!(len, list.length);
}

#[test]
fn test_basic() {
    let mut m = List::new::<~int>();
    assert_eq!(m.pop_front(), None);
    assert_eq!(m.pop_back(), None);
    assert_eq!(m.pop_front(), None);
    m.push_front(~1);
    assert_eq!(m.pop_front(), Some(~1));
    m.push_back(~2);
    m.push_back(~3);
    assert_eq!(m.len(), 2);
    assert_eq!(m.pop_front(), Some(~2));
    assert_eq!(m.pop_front(), Some(~3));
    assert_eq!(m.len(), 0);
    assert_eq!(m.pop_front(), None);
    m.push_back(~1);
    m.push_back(~3);
    m.push_back(~5);
    m.push_back(~7);
    assert_eq!(m.pop_front(), Some(~1));

    let mut n = List::new();
    n.push_front(2);
    n.push_front(3);
    {
        assert_eq!(n.peek_front().unwrap(), &3);
        let x = n.peek_front_mut().unwrap();
        assert_eq!(*x, 3);
        *x = 0;
    }
    {
        assert_eq!(n.peek_back().unwrap(), &2);
        let y = n.peek_back_mut().unwrap();
        assert_eq!(*y, 2);
        *y = 1;
    }
    assert_eq!(n.pop_front(), Some(0));
    assert_eq!(n.pop_front(), Some(1));
}

#[cfg(test)]
fn generate_test() -> List<int> {
    list_from(&[0,1,2,3,4,5,6])
}

#[cfg(test)]
fn list_from<T: Copy>(v: &[T]) -> List<T> {
    v.iter().transform(|x| copy *x).collect()
}

#[test]
fn test_append() {
    {
        let mut m = List::new();
        let mut n = List::new();
        n.push_back(2);
        m.append(n);
        assert_eq!(m.len(), 1);
        assert_eq!(m.pop_back(), Some(2));
        check_links(&m);
    }
    {
        let mut m = List::new();
        let n = List::new();
        m.push_back(2);
        m.append(n);
        assert_eq!(m.len(), 1);
        assert_eq!(m.pop_back(), Some(2));
        check_links(&m);
    }

    let v = ~[1,2,3,4,5];
    let u = ~[9,8,1,2,3,4,5];
    let mut m = list_from(v);
    m.append(list_from(u));
    check_links(&m);
    let sum = v + u;
    assert_eq!(sum.len(), m.len());
    for sum.consume_iter().advance |elt| {
        assert_eq!(m.pop_front(), Some(elt))
    }
}

#[test]
fn test_prepend() {
    {
        let mut m = List::new();
        let mut n = List::new();
        n.push_back(2);
        m.prepend(n);
        assert_eq!(m.len(), 1);
        assert_eq!(m.pop_back(), Some(2));
        check_links(&m);
    }

    let v = ~[1,2,3,4,5];
    let u = ~[9,8,1,2,3,4,5];
    let mut m = list_from(v);
    m.prepend(list_from(u));
    check_links(&m);
    let sum = u + v;
    assert_eq!(sum.len(), m.len());
    for sum.consume_iter().advance |elt| {
        assert_eq!(m.pop_front(), Some(elt))
    }
}

#[test]
fn test_iterator() {
    let m = generate_test();
    for m.iter().enumerate().advance |(i, elt)| {
        assert_eq!(i as int, *elt);
    }
    let mut n = List::new();
    assert_eq!(n.iter().next(), None);
    n.push_front(4);
    let mut it = n.iter();
    assert_eq!(it.next().unwrap(), &4);
    assert_eq!(it.next(), None);
}

#[test]
fn test_rev_iter() {
    let m = generate_test();
    for m.rev_iter().enumerate().advance |(i, elt)| {
        assert_eq!((6 - i) as int, *elt);
    }
    let mut n = List::new();
    assert_eq!(n.rev_iter().next(), None);
    n.push_front(4);
    let mut it = n.rev_iter();
    assert_eq!(it.next().unwrap(), &4);
    assert_eq!(it.next(), None);
}

#[test]
fn test_mut_iter() {
    let mut m = generate_test();
    let mut len = m.len();
    for m.mut_iter().enumerate().advance |(i, elt)| {
        assert_eq!(i as int, *elt);
        len -= 1;
    }
    assert_eq!(len, 0);
    let mut n = List::new();
    assert!(n.mut_iter().next().is_none());
    n.push_front(4);
    let mut it = n.mut_iter();
    assert!(it.next().is_some());
    assert!(it.next().is_none());
}

#[test]
fn test_list_cursor() {
    let mut m = generate_test();
    let len = m.len();
    {
        let mut it = m.mut_iter();
        loop {
            match it.next() {
                None => break,
                Some(elt) => it.insert_before(*elt * 2),
            }
        }
    }
    assert_eq!(m.len(), len * 2);
    check_links(&m);
}

#[test]
fn test_merge() {
    let mut m = list_from([0, 1, 3, 5, 6, 7, 2]);
    let n = list_from([-1, 0, 0, 7, 7, 9]);
    let len = m.len() + n.len();
    m.merge(n, |a, b| a <= b);
    assert_eq!(m.len(), len);
    check_links(&m);
    let res = m.consume_iter().collect::<~[int]>();
    assert_eq!(res, ~[-1, 0, 0, 1, 0, 3, 5, 6, 7, 2, 7, 7, 9]);
}

#[test]
fn test_insert_ordered() {
    let mut n = List::new();
    n.insert_ordered(1);
    assert_eq!(n.len(), 1);
    assert_eq!(n.pop_front(), Some(1));

    let mut m = List::new();
    m.push_back(2);
    m.push_back(4);
    m.insert_ordered(3);
    check_links(&m);
    assert_eq!(~[2,3,4], m.consume_iter().collect::<~[int]>());
}

#[test]
fn test_mut_rev_iter() {
    let mut m = generate_test();
    for m.mut_rev_iter().enumerate().advance |(i, elt)| {
        assert_eq!((6-i) as int, *elt);
    }
    let mut n = List::new();
    assert!(n.mut_rev_iter().next().is_none());
    n.push_front(4);
    let mut it = n.mut_rev_iter();
    assert!(it.next().is_some());
    assert!(it.next().is_none());
}

#[test]
fn test_send() {
    let n = list_from([1,2,3]);
    do spawn {
        check_links(&n);
        assert_eq!(~[&1,&2,&3], n.iter().collect::<~[&int]>());
    }
}

#[test]
fn test_eq() {
    let mut n: List<u8> = list_from([]);
    let mut m = list_from([]);
    assert_eq!(&n, &m);
    n.push_front(1);
    assert!(n != m);
    m.push_back(1);
    assert_eq!(&n, &m);
}

#[test]
fn test_fuzz() {
    for 25.times {
        fuzz_test(3);
        fuzz_test(16);
        fuzz_test(189);
    }
}

#[cfg(test)]
fn fuzz_test(sz: int) {
    use std::rand;
    use std::int;

    let mut m = List::new::<int>();
    let mut v = ~[];
    for int::range(0i, sz) |i| {
        check_links(&m);
        let r: u8 = rand::random();
        match r % 6 {
            0 => {
                m.pop_back();
                if v.len() > 0 { v.pop(); }
            }
            1 => {
                m.pop_front();
                if v.len() > 0 { v.shift(); }
            }
            2 | 4 =>  {
                m.push_front(-i);
                v.unshift(-i);
            }
            3 | 5 | _ => {
                m.push_back(i);
                v.push(i);
            }
        }
    }

    check_links(&m);

    let mut i = 0u;
    for m.consume_iter().zip(v.iter()).advance |(a, &b)| {
        i += 1;
        assert_eq!(a, b);
    }
    assert_eq!(i, v.len());
}

#[cfg(test)]
mod test_bench {
    use extra::test;

    use super::*;

    #[bench]
    fn bench_collect_into(b: &mut test::BenchHarness) {
        let v = &[0, ..64];
        do b.iter {
            let _: List<int> = v.iter().transform(|&x|x).collect();
        }
    }
    #[bench]
    fn bench_collect_into_vec(b: &mut test::BenchHarness) {
        let v = &[0, ..64];
        do b.iter {
            let _: ~[int] = v.iter().transform(|&x|x).collect();
        }
    }

    #[bench]
    fn bench_push_front(b: &mut test::BenchHarness) {
        let mut m = List::new::<int>();
        do b.iter {
            m.push_front(0);
        }
    }
    #[bench]
    fn bench_push_front_vec_size10(b: &mut test::BenchHarness) {
        let mut m = ~[0, ..10];
        do b.iter {
            m.unshift(0);
            m.pop(); // to keep it fair, dont' grow the vec
        }
    }

    #[bench]
    fn bench_push_back(b: &mut test::BenchHarness) {
        let mut m = List::new::<int>();
        do b.iter {
            m.push_back(0);
        }
    }
    #[bench]
    fn bench_push_back_vec(b: &mut test::BenchHarness) {
        let mut m = ~[];
        do b.iter {
            m.push(0);
        }
    }

    #[bench]
    fn bench_push_back_pop_back(b: &mut test::BenchHarness) {
        let mut m = List::new::<int>();
        do b.iter {
            m.push_back(0);
            m.pop_back();
        }
    }
    #[bench]
    fn bench_push_back_pop_back_vec(b: &mut test::BenchHarness) {
        let mut m = ~[];
        do b.iter {
            m.push(0);
            m.pop();
        }
    }

    #[bench]
    fn bench_iter(b: &mut test::BenchHarness) {
        let v = &[0, ..128];
        let m: List<int> = v.iter().transform(|&x|x).collect();
        do b.iter {
            for m.iter().advance |_| {}
        }
    }
    #[bench]
    fn bench_iter_mut(b: &mut test::BenchHarness) {
        let v = &[0, ..128];
        let mut m: List<int> = v.iter().transform(|&x|x).collect();
        do b.iter {
            for m.mut_iter().advance |_| {}
        }
    }
    #[bench]
    fn bench_iter_rev(b: &mut test::BenchHarness) {
        let v = &[0, ..128];
        let m: List<int> = v.iter().transform(|&x|x).collect();
        do b.iter {
            for m.rev_iter().advance |_| {}
        }
    }
    #[bench]
    fn bench_iter_mut_rev(b: &mut test::BenchHarness) {
        let v = &[0, ..128];
        let mut m: List<int> = v.iter().transform(|&x|x).collect();
        do b.iter {
            for m.mut_rev_iter().advance |_| {}
        }
    }
    #[bench]
    fn bench_iter_vec(b: &mut test::BenchHarness) {
        let v = &[0, ..128];
        do b.iter {
            for v.iter().advance |_| {}
        }
    }
}

