use core::iter::*;

#[test]
fn test_inspect() {
    let xs = [1, 2, 3, 4];
    let mut n = 0;

    let ys = xs.iter().cloned().inspect(|_| n += 1).collect::<Vec<usize>>();

    assert_eq!(n, xs.len());
    assert_eq!(&xs[..], &ys[..]);
}

#[test]
fn test_inspect_fold() {
    let xs = [1, 2, 3, 4];
    let mut n = 0;
    {
        let it = xs.iter().inspect(|_| n += 1);
        let i = it.fold(0, |i, &x| {
            assert_eq!(x, xs[i]);
            i + 1
        });
        assert_eq!(i, xs.len());
    }
    assert_eq!(n, xs.len());

    let mut n = 0;
    {
        let it = xs.iter().inspect(|_| n += 1);
        let i = it.rfold(xs.len(), |i, &x| {
            assert_eq!(x, xs[i - 1]);
            i - 1
        });
        assert_eq!(i, 0);
    }
    assert_eq!(n, xs.len());
}
