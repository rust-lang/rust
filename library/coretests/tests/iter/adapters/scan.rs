use core::iter::*;

#[test]
fn test_iterator_scan() {
    // test the type inference
    fn add(old: &mut isize, new: &usize) -> Option<f64> {
        *old += *new as isize;
        Some(*old as f64)
    }
    let xs = [0, 1, 2, 3, 4];
    let ys = [0f64, 1.0, 3.0, 6.0, 10.0];

    let it = xs.iter().scan(0, add);
    let mut i = 0;
    for x in it {
        assert_eq!(x, ys[i]);
        i += 1;
    }
    assert_eq!(i, ys.len());
}
