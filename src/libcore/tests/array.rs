use core::array::{FixedSizeArray, IntoIter};
use core::convert::TryFrom;

#[test]
fn fixed_size_array() {
    let mut array = [0; 64];
    let mut zero_sized = [(); 64];
    let mut empty_array = [0; 0];
    let mut empty_zero_sized = [(); 0];

    assert_eq!(FixedSizeArray::as_slice(&array).len(), 64);
    assert_eq!(FixedSizeArray::as_slice(&zero_sized).len(), 64);
    assert_eq!(FixedSizeArray::as_slice(&empty_array).len(), 0);
    assert_eq!(FixedSizeArray::as_slice(&empty_zero_sized).len(), 0);

    assert_eq!(FixedSizeArray::as_mut_slice(&mut array).len(), 64);
    assert_eq!(FixedSizeArray::as_mut_slice(&mut zero_sized).len(), 64);
    assert_eq!(FixedSizeArray::as_mut_slice(&mut empty_array).len(), 0);
    assert_eq!(FixedSizeArray::as_mut_slice(&mut empty_zero_sized).len(), 0);
}

#[test]
fn array_try_from() {
    macro_rules! test {
        ($($N:expr)+) => {
            $({
                type Array = [u8; $N];
                let array: Array = [0; $N];
                let slice: &[u8] = &array[..];

                let result = <&Array>::try_from(slice);
                assert_eq!(&array, result.unwrap());
            })+
        }
    }
    test! {
         0  1  2  3  4  5  6  7  8  9
        10 11 12 13 14 15 16 17 18 19
        20 21 22 23 24 25 26 27 28 29
        30 31 32
    }
}


#[test]
fn iterator_collect() {
    let arr = [0, 1, 2, 5, 9];
    let v: Vec<_> = IntoIter::new(arr.clone()).collect();
    assert_eq!(&arr[..], &v[..]);
}

#[test]
fn iterator_rev_collect() {
    let arr = [0, 1, 2, 5, 9];
    let v: Vec<_> = IntoIter::new(arr.clone()).rev().collect();
    assert_eq!(&v[..], &[9, 5, 2, 1, 0]);
}

#[test]
fn iterator_nth() {
    let v = [0, 1, 2, 3, 4];
    for i in 0..v.len() {
        assert_eq!(IntoIter::new(v.clone()).nth(i).unwrap(), v[i]);
    }
    assert_eq!(IntoIter::new(v.clone()).nth(v.len()), None);

    let mut iter = IntoIter::new(v);
    assert_eq!(iter.nth(2).unwrap(), v[2]);
    assert_eq!(iter.nth(1).unwrap(), v[4]);
}

#[test]
fn iterator_last() {
    let v = [0, 1, 2, 3, 4];
    assert_eq!(IntoIter::new(v).last().unwrap(), 4);
    assert_eq!(IntoIter::new([0]).last().unwrap(), 0);

    let mut it = IntoIter::new([0, 9, 2, 4]);
    assert_eq!(it.next_back(), Some(4));
    assert_eq!(it.last(), Some(2));
}

#[test]
fn iterator_clone() {
    let mut it = IntoIter::new([0, 2, 4, 6, 8]);
    assert_eq!(it.next(), Some(0));
    assert_eq!(it.next_back(), Some(8));
    let mut clone = it.clone();
    assert_eq!(it.next_back(), Some(6));
    assert_eq!(clone.next_back(), Some(6));
    assert_eq!(it.next_back(), Some(4));
    assert_eq!(clone.next_back(), Some(4));
    assert_eq!(it.next(), Some(2));
    assert_eq!(clone.next(), Some(2));
}

#[test]
fn iterator_fused() {
    let mut it = IntoIter::new([0, 9, 2]);
    assert_eq!(it.next(), Some(0));
    assert_eq!(it.next(), Some(9));
    assert_eq!(it.next(), Some(2));
    assert_eq!(it.next(), None);
    assert_eq!(it.next(), None);
    assert_eq!(it.next(), None);
    assert_eq!(it.next(), None);
    assert_eq!(it.next(), None);
}

#[test]
fn iterator_len() {
    let mut it = IntoIter::new([0, 1, 2, 5, 9]);
    assert_eq!(it.size_hint(), (5, Some(5)));
    assert_eq!(it.len(), 5);
    assert_eq!(it.is_empty(), false);

    assert_eq!(it.next(), Some(0));
    assert_eq!(it.size_hint(), (4, Some(4)));
    assert_eq!(it.len(), 4);
    assert_eq!(it.is_empty(), false);

    assert_eq!(it.next_back(), Some(9));
    assert_eq!(it.size_hint(), (3, Some(3)));
    assert_eq!(it.len(), 3);
    assert_eq!(it.is_empty(), false);

    // Empty
    let it = IntoIter::new([] as [String; 0]);
    assert_eq!(it.size_hint(), (0, Some(0)));
    assert_eq!(it.len(), 0);
    assert_eq!(it.is_empty(), true);
}

#[test]
fn iterator_count() {
    let v = [0, 1, 2, 3, 4];
    assert_eq!(IntoIter::new(v.clone()).count(), 5);

    let mut iter2 = IntoIter::new(v);
    iter2.next();
    iter2.next();
    assert_eq!(iter2.count(), 3);
}

#[test]
fn iterator_flat_map() {
    assert!((0..5).flat_map(|i| IntoIter::new([2 * i, 2 * i + 1])).eq(0..10));
}

#[test]
fn iterator_debug() {
    let arr = [0, 1, 2, 5, 9];
    assert_eq!(
        format!("{:?}", IntoIter::new(arr)),
        "IntoIter([0, 1, 2, 5, 9])",
    );
}

#[test]
fn iterator_drops() {
    use core::cell::Cell;

    // This test makes sure the correct number of elements are dropped. The `R`
    // type is just a reference to a `Cell` that is incremented when an `R` is
    // dropped.

    #[derive(Clone)]
    struct Foo<'a>(&'a Cell<usize>);

    impl Drop for Foo<'_> {
       fn drop(&mut self) {
            self.0.set(self.0.get() + 1);
        }
    }

    fn five(i: &Cell<usize>) -> [Foo<'_>; 5] {
        // This is somewhat verbose because `Foo` does not implement `Copy`
        // since it implements `Drop`. Consequently, we cannot write
        // `[Foo(i); 5]`.
        [Foo(i), Foo(i), Foo(i), Foo(i), Foo(i)]
    }

    // Simple: drop new iterator.
    let i = Cell::new(0);
    {
        IntoIter::new(five(&i));
    }
    assert_eq!(i.get(), 5);

    // Call `next()` once.
    let i = Cell::new(0);
    {
        let mut iter = IntoIter::new(five(&i));
        let _x = iter.next();
        assert_eq!(i.get(), 0);
        assert_eq!(iter.count(), 4);
        assert_eq!(i.get(), 4);
    }
    assert_eq!(i.get(), 5);

    // Check `clone` and calling `next`/`next_back`.
    let i = Cell::new(0);
    {
        let mut iter = IntoIter::new(five(&i));
        iter.next();
        assert_eq!(i.get(), 1);
        iter.next_back();
        assert_eq!(i.get(), 2);

        let mut clone = iter.clone();
        assert_eq!(i.get(), 2);

        iter.next();
        assert_eq!(i.get(), 3);

        clone.next();
        assert_eq!(i.get(), 4);

        assert_eq!(clone.count(), 2);
        assert_eq!(i.get(), 6);
    }
    assert_eq!(i.get(), 8);

    // Check via `nth`.
    let i = Cell::new(0);
    {
        let mut iter = IntoIter::new(five(&i));
        let _x = iter.nth(2);
        assert_eq!(i.get(), 2);
        let _y = iter.last();
        assert_eq!(i.get(), 3);
    }
    assert_eq!(i.get(), 5);

    // Check every element.
    let i = Cell::new(0);
    for (index, _x) in IntoIter::new(five(&i)).enumerate() {
        assert_eq!(i.get(), index);
    }
    assert_eq!(i.get(), 5);

    let i = Cell::new(0);
    for (index, _x) in IntoIter::new(five(&i)).rev().enumerate() {
        assert_eq!(i.get(), index);
    }
    assert_eq!(i.get(), 5);
}
