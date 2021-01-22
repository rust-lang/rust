use core::iter::*;

#[test]
fn test_filter_map() {
    let it = (0..).step_by(1).take(10).filter_map(|x| if x % 2 == 0 { Some(x * x) } else { None });
    assert_eq!(it.collect::<Vec<usize>>(), [0 * 0, 2 * 2, 4 * 4, 6 * 6, 8 * 8]);
}

#[test]
fn test_filter_map_fold() {
    let xs = [0, 1, 2, 3, 4, 5, 6, 7, 8];
    let ys = [0 * 0, 2 * 2, 4 * 4, 6 * 6, 8 * 8];
    let it = xs.iter().filter_map(|&x| if x % 2 == 0 { Some(x * x) } else { None });
    let i = it.fold(0, |i, x| {
        assert_eq!(x, ys[i]);
        i + 1
    });
    assert_eq!(i, ys.len());

    let it = xs.iter().filter_map(|&x| if x % 2 == 0 { Some(x * x) } else { None });
    let i = it.rfold(ys.len(), |i, x| {
        assert_eq!(x, ys[i - 1]);
        i - 1
    });
    assert_eq!(i, 0);
}
