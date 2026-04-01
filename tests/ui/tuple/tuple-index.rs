//@ run-pass

struct Point(isize, isize);

fn main() {
    let mut x = Point(3, 2);
    assert_eq!(x.0, 3);
    assert_eq!(x.1, 2);
    x.0 += 5;
    assert_eq!(x.0, 8);
    {
        let ry = &mut x.1;
        *ry -= 2;
        x.0 += 3;
        assert_eq!(x.0, 11);
    }
    assert_eq!(x.1, 0);

    let mut x = (3, 2);
    assert_eq!(x.0, 3);
    assert_eq!(x.1, 2);
    x.0 += 5;
    assert_eq!(x.0, 8);
    {
        let ry = &mut x.1;
        *ry -= 2;
        x.0 += 3;
        assert_eq!(x.0, 11);
    }
    assert_eq!(x.1, 0);

}
