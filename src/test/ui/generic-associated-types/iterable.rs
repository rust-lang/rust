#![allow(incomplete_features)]
#![feature(generic_associated_types)]

// run-pass

trait Iterable {
    type Item<'a> where Self: 'a;
    type Iter<'a>: Iterator<Item = Self::Item<'a>> where Self: 'a;

    fn iter<'a>(&'a self) -> Self::Iter<'a>;
}

// Impl for struct type
impl<T> Iterable for Vec<T> {
    type Item<'a> where T: 'a = <std::slice::Iter<'a, T> as Iterator>::Item;
    type Iter<'a> where T: 'a = std::slice::Iter<'a, T>;

    fn iter<'a>(&'a self) -> Self::Iter<'a> {
        self[..].iter()
    }
}

// Impl for a primitive type
impl<T> Iterable for [T] {
    type Item<'a> where T: 'a = <std::slice::Iter<'a, T> as Iterator>::Item;
    type Iter<'a> where T: 'a = std::slice::Iter<'a, T>;

    fn iter<'a>(&'a self) -> Self::Iter<'a> {
        self.iter()
    }
}

fn make_iter<'a, I: Iterable + ?Sized>(it: &'a I) -> I::Iter<'a> {
    it.iter()
}

fn get_first<'a, I: Iterable + ?Sized>(it: &'a I) -> Option<I::Item<'a>> {
    it.iter().next()
}

fn main() {
    let v = vec![1, 2, 3];
    assert_eq!(v, make_iter(&v).copied().collect::<Vec<_>>());
    assert_eq!(v, make_iter(&*v).copied().collect::<Vec<_>>());
    assert_eq!(Some(&1), get_first(&v));
    assert_eq!(Some(&1), get_first(&*v));
}
