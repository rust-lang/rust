use core::iter::*;
use std::num::Saturating;

#[test]
fn test_iterator_sum() {
    let v: &[i32] = &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    assert_eq!(v[..4].iter().cloned().sum::<i32>(), 6);
    assert_eq!(v.iter().cloned().sum::<i32>(), 55);
    assert_eq!(v[..0].iter().cloned().sum::<i32>(), 0);
}

#[test]
fn test_iterator_sum_result() {
    let v: &[Result<i32, ()>] = &[Ok(1), Ok(2), Ok(3), Ok(4)];
    assert_eq!(v.iter().cloned().sum::<Result<i32, _>>(), Ok(10));
    let v: &[Result<i32, ()>] = &[Ok(1), Err(()), Ok(3), Ok(4)];
    assert_eq!(v.iter().cloned().sum::<Result<i32, _>>(), Err(()));

    #[derive(PartialEq, Debug)]
    struct S(Result<i32, ()>);

    impl Sum<Result<i32, ()>> for S {
        fn sum<I: Iterator<Item = Result<i32, ()>>>(mut iter: I) -> Self {
            // takes the sum by repeatedly calling `next` on `iter`,
            // thus testing that repeated calls to `ResultShunt::try_fold`
            // produce the expected results
            Self(iter.by_ref().sum())
        }
    }

    let v: &[Result<i32, ()>] = &[Ok(1), Ok(2), Ok(3), Ok(4)];
    assert_eq!(v.iter().cloned().sum::<S>(), S(Ok(10)));
    let v: &[Result<i32, ()>] = &[Ok(1), Err(()), Ok(3), Ok(4)];
    assert_eq!(v.iter().cloned().sum::<S>(), S(Err(())));
}

#[test]
fn test_iterator_sum_option() {
    let v: &[Option<i32>] = &[Some(1), Some(2), Some(3), Some(4)];
    assert_eq!(v.iter().cloned().sum::<Option<i32>>(), Some(10));
    let v: &[Option<i32>] = &[Some(1), None, Some(3), Some(4)];
    assert_eq!(v.iter().cloned().sum::<Option<i32>>(), None);
}

#[test]
fn test_iterator_product() {
    let v: &[i32] = &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    assert_eq!(v[..4].iter().cloned().product::<i32>(), 0);
    assert_eq!(v[1..5].iter().cloned().product::<i32>(), 24);
    assert_eq!(v[..0].iter().cloned().product::<i32>(), 1);
}

#[test]
fn test_iterator_product_result() {
    let v: &[Result<i32, ()>] = &[Ok(1), Ok(2), Ok(3), Ok(4)];
    assert_eq!(v.iter().cloned().product::<Result<i32, _>>(), Ok(24));
    let v: &[Result<i32, ()>] = &[Ok(1), Err(()), Ok(3), Ok(4)];
    assert_eq!(v.iter().cloned().product::<Result<i32, _>>(), Err(()));
}

#[test]
fn test_iterator_product_option() {
    let v: &[Option<i32>] = &[Some(1), Some(2), Some(3), Some(4)];
    assert_eq!(v.iter().cloned().product::<Option<i32>>(), Some(24));
    let v: &[Option<i32>] = &[Some(1), None, Some(3), Some(4)];
    assert_eq!(v.iter().cloned().product::<Option<i32>>(), None);
}

#[test]
fn test_saturating_sum_product() {
    let v = (1u32..=10).map(|i| Saturating(i));
    assert_eq!(v.sum::<Saturating<u32>>(), Saturating(55));
    let v = (1u32..=10).map(|i| Saturating(i));
    assert_eq!(v.product::<Saturating<u32>>(), Saturating(3628800));
    let v = [Saturating(usize::MAX), Saturating(2)];
    assert_eq!(v.iter().copied().sum::<Saturating<usize>>(), Saturating(usize::MAX));
    assert_eq!(v.iter().copied().product::<Saturating<usize>>(), Saturating(usize::MAX));

    let mut cnt = 0;
    let v = 250..=u8::MAX;
    assert_eq!(
        v.map(|i| {
            cnt += 1;
            Saturating(i)
        })
        .sum::<Saturating<u8>>(),
        Saturating(u8::MAX)
    );
    assert_eq!(cnt, 6); // no short-circuiting

    let mut cnt = 0;
    let v = (250..=u8::MAX).chain(0..5);
    assert_eq!(
        v.map(|i| {
            cnt += 1;
            Saturating(i)
        })
        .product::<Saturating<u8>>(),
        Saturating(0)
    );
    assert_eq!(cnt, 11); // no short-circuiting
}
