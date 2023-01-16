use core::iter::*;
use core::num::NonZero;

#[test]
fn test_iterator_take() {
    let xs = [0, 1, 2, 3, 5, 13, 15, 16, 17, 19];
    let ys = [0, 1, 2, 3, 5];

    let mut it = xs.iter().take(ys.len());
    let mut i = 0;
    assert_eq!(it.len(), ys.len());
    while let Some(&x) = it.next() {
        assert_eq!(x, ys[i]);
        i += 1;
        assert_eq!(it.len(), ys.len() - i);
    }
    assert_eq!(i, ys.len());
    assert_eq!(it.len(), 0);

    let mut it = xs.iter().take(ys.len());
    let mut i = 0;
    assert_eq!(it.len(), ys.len());
    while let Some(&x) = it.next_back() {
        i += 1;
        assert_eq!(x, ys[ys.len() - i]);
        assert_eq!(it.len(), ys.len() - i);
    }
    assert_eq!(i, ys.len());
    assert_eq!(it.len(), 0);
}

#[test]
fn test_iterator_take_nth() {
    let xs = [0, 1, 2, 4, 5];
    let mut it = xs.iter();
    {
        let mut take = it.by_ref().take(3);
        let mut i = 0;
        while let Some(&x) = take.nth(0) {
            assert_eq!(x, i);
            i += 1;
        }
    }
    assert_eq!(it.nth(1), Some(&5));
    assert_eq!(it.nth(0), None);

    let xs = [0, 1, 2, 3, 4];
    let mut it = xs.iter().take(7);
    let mut i = 1;
    while let Some(&x) = it.nth(1) {
        assert_eq!(x, i);
        i += 2;
    }
}

#[test]
fn test_iterator_take_nth_back() {
    let xs = [0, 1, 2, 4, 5];
    let mut it = xs.iter();
    {
        let mut take = it.by_ref().take(3);
        let mut i = 0;
        while let Some(&x) = take.nth_back(0) {
            i += 1;
            assert_eq!(x, 3 - i);
        }
    }
    assert_eq!(it.nth_back(0), None);

    let xs = [0, 1, 2, 3, 4];
    let mut it = xs.iter().take(7);
    assert_eq!(it.nth_back(1), Some(&3));
    assert_eq!(it.nth_back(1), Some(&1));
    assert_eq!(it.nth_back(1), None);
}

#[test]
fn test_take_advance_by() {
    let mut take = (0..10).take(3);
    assert_eq!(take.advance_by(2), Ok(()));
    assert_eq!(take.next(), Some(2));
    assert_eq!(take.advance_by(1), Err(NonZero::new(1).unwrap()));

    assert_eq!((0..0).take(10).advance_by(0), Ok(()));
    assert_eq!((0..0).take(10).advance_by(1), Err(NonZero::new(1).unwrap()));
    assert_eq!((0..10).take(4).advance_by(5), Err(NonZero::new(1).unwrap()));

    let mut take = (0..10).take(3);
    assert_eq!(take.advance_back_by(2), Ok(()));
    assert_eq!(take.next(), Some(0));
    assert_eq!(take.advance_back_by(1), Err(NonZero::new(1).unwrap()));

    assert_eq!((0..2).take(1).advance_back_by(10), Err(NonZero::new(9).unwrap()));
    assert_eq!((0..0).take(1).advance_back_by(1), Err(NonZero::new(1).unwrap()));
    assert_eq!((0..0).take(1).advance_back_by(0), Ok(()));
    assert_eq!(
        (0..usize::MAX).take(100).advance_back_by(usize::MAX),
        Err(NonZero::new(usize::MAX - 100).unwrap())
    );
}

#[test]
fn test_iterator_take_short() {
    let xs = [0, 1, 2, 3];

    let mut it = xs.iter().take(5);
    let mut i = 0;
    assert_eq!(it.len(), xs.len());
    while let Some(&x) = it.next() {
        assert_eq!(x, xs[i]);
        i += 1;
        assert_eq!(it.len(), xs.len() - i);
    }
    assert_eq!(i, xs.len());
    assert_eq!(it.len(), 0);

    let mut it = xs.iter().take(5);
    let mut i = 0;
    assert_eq!(it.len(), xs.len());
    while let Some(&x) = it.next_back() {
        i += 1;
        assert_eq!(x, xs[xs.len() - i]);
        assert_eq!(it.len(), xs.len() - i);
    }
    assert_eq!(i, xs.len());
    assert_eq!(it.len(), 0);
}

#[test]
fn test_take_try_folds() {
    let f = &|acc, x| i32::checked_add(2 * acc, x);
    assert_eq!((10..30).take(10).try_fold(7, f), (10..20).try_fold(7, f));
    assert_eq!((10..30).take(10).try_rfold(7, f), (10..20).try_rfold(7, f));

    let mut iter = (10..30).take(20);
    assert_eq!(iter.try_fold(0, i8::checked_add), None);
    assert_eq!(iter.next(), Some(20));
    assert_eq!(iter.try_rfold(0, i8::checked_add), None);
    assert_eq!(iter.next_back(), Some(24));

    let mut iter = (2..20).take(3);
    assert_eq!(iter.try_for_each(Err), Err(2));
    assert_eq!(iter.try_for_each(Err), Err(3));
    assert_eq!(iter.try_for_each(Err), Err(4));
    assert_eq!(iter.try_for_each(Err), Ok(()));

    let mut iter = (2..20).take(3).rev();
    assert_eq!(iter.try_for_each(Err), Err(4));
    assert_eq!(iter.try_for_each(Err), Err(3));
    assert_eq!(iter.try_for_each(Err), Err(2));
    assert_eq!(iter.try_for_each(Err), Ok(()));
}

#[test]
fn test_byref_take_consumed_items() {
    let mut inner = 10..90;

    let mut count = 0;
    inner.by_ref().take(0).for_each(|_| count += 1);
    assert_eq!(count, 0);
    assert_eq!(inner, 10..90);

    let mut count = 0;
    inner.by_ref().take(10).for_each(|_| count += 1);
    assert_eq!(count, 10);
    assert_eq!(inner, 20..90);

    let mut count = 0;
    inner.by_ref().take(100).for_each(|_| count += 1);
    assert_eq!(count, 70);
    assert_eq!(inner, 90..90);
}

#[test]
fn test_exact_size_take_repeat() {
    let mut iter = core::iter::repeat(42).take(40);
    assert_eq!((40, Some(40)), iter.size_hint());
    assert_eq!(40, iter.len());

    assert_eq!(Some(42), iter.next());
    assert_eq!((39, Some(39)), iter.size_hint());
    assert_eq!(39, iter.len());

    assert_eq!(Some(42), iter.next_back());
    assert_eq!((38, Some(38)), iter.size_hint());
    assert_eq!(38, iter.len());

    assert_eq!(Some(42), iter.nth(3));
    assert_eq!((34, Some(34)), iter.size_hint());
    assert_eq!(34, iter.len());

    assert_eq!(Some(42), iter.nth_back(3));
    assert_eq!((30, Some(30)), iter.size_hint());
    assert_eq!(30, iter.len());

    assert_eq!(Ok(()), iter.advance_by(10));
    assert_eq!((20, Some(20)), iter.size_hint());
    assert_eq!(20, iter.len());

    assert_eq!(Ok(()), iter.advance_back_by(10));
    assert_eq!((10, Some(10)), iter.size_hint());
    assert_eq!(10, iter.len());
}

#[test]
fn test_exact_size_take_repeat_with() {
    let mut counter = 0;
    let mut iter = core::iter::repeat_with(move || {
        counter += 1;
        counter
    })
    .take(40);
    assert_eq!((40, Some(40)), iter.size_hint());
    assert_eq!(40, iter.len());

    assert_eq!(Some(1), iter.next());
    assert_eq!((39, Some(39)), iter.size_hint());
    assert_eq!(39, iter.len());

    assert_eq!(Some(5), iter.nth(3));
    assert_eq!((35, Some(35)), iter.size_hint());
    assert_eq!(35, iter.len());

    assert_eq!(Ok(()), iter.advance_by(10));
    assert_eq!((25, Some(25)), iter.size_hint());
    assert_eq!(25, iter.len());

    assert_eq!(Some(16), iter.next());
    assert_eq!((24, Some(24)), iter.size_hint());
    assert_eq!(24, iter.len());
}

// This is https://github.com/rust-lang/rust/issues/104729 with all uses of
// repeat(0) were replaced by repeat(0).take(20).
#[test]
fn test_reverse_on_zip() {
    let vec_1 = [1; 10];

    let zipped_iter = vec_1.iter().copied().zip(core::iter::repeat(0).take(20));

    // Forward
    for (one, zero) in zipped_iter {
        assert_eq!((1, 0), (one, zero));
    }

    let rev_vec_iter = vec_1.iter().rev();
    let rev_repeat_iter = std::iter::repeat(0).take(20).rev();

    // Manual reversed zip
    let rev_zipped_iter = rev_vec_iter.zip(rev_repeat_iter);

    for (&one, zero) in rev_zipped_iter {
        assert_eq!((1, 0), (one, zero));
    }

    let zipped_iter = vec_1.iter().zip(core::iter::repeat(0).take(20));

    // Cannot call rev here for automatic reversed zip constuction
    for (&one, zero) in zipped_iter.rev() {
        assert_eq!((1, 0), (one, zero));
    }
}
