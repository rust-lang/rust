// run-pass

// Test slicing expressions on slices and Vecs.


fn main() {
    let x: &[isize] = &[1, 2, 3, 4, 5];
    let cmp: &[isize] = &[1, 2, 3, 4, 5];
    assert_eq!(&x[..], cmp);
    let cmp: &[isize] = &[3, 4, 5];
    assert_eq!(&x[2..], cmp);
    let cmp: &[isize] = &[1, 2, 3];
    assert_eq!(&x[..3], cmp);
    let cmp: &[isize] = &[2, 3, 4];
    assert_eq!(&x[1..4], cmp);

    let x: Vec<isize> = vec![1, 2, 3, 4, 5];
    let cmp: &[isize] = &[1, 2, 3, 4, 5];
    assert_eq!(&x[..], cmp);
    let cmp: &[isize] = &[3, 4, 5];
    assert_eq!(&x[2..], cmp);
    let cmp: &[isize] = &[1, 2, 3];
    assert_eq!(&x[..3], cmp);
    let cmp: &[isize] = &[2, 3, 4];
    assert_eq!(&x[1..4], cmp);

    let x: &mut [isize] = &mut [1, 2, 3, 4, 5];
    {
        let cmp: &mut [isize] = &mut [1, 2, 3, 4, 5];
        assert_eq!(&mut x[..], cmp);
    }
    {
        let cmp: &mut [isize] = &mut [3, 4, 5];
        assert_eq!(&mut x[2..], cmp);
    }
    {
        let cmp: &mut [isize] = &mut [1, 2, 3];
        assert_eq!(&mut x[..3], cmp);
    }
    {
        let cmp: &mut [isize] = &mut [2, 3, 4];
        assert_eq!(&mut x[1..4], cmp);
    }

    let mut x: Vec<isize> = vec![1, 2, 3, 4, 5];
    {
        let cmp: &mut [isize] = &mut [1, 2, 3, 4, 5];
        assert_eq!(&mut x[..], cmp);
    }
    {
        let cmp: &mut [isize] = &mut [3, 4, 5];
        assert_eq!(&mut x[2..], cmp);
    }
    {
        let cmp: &mut [isize] = &mut [1, 2, 3];
        assert_eq!(&mut x[..3], cmp);
    }
    {
        let cmp: &mut [isize] = &mut [2, 3, 4];
        assert_eq!(&mut x[1..4], cmp);
    }
}
