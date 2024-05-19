//@ run-pass

struct Foo(isize, isize);

fn main() {
    let x = (1, 2);
    let a = &x.0;
    let b = &x.0;
    assert_eq!(*a, 1);
    assert_eq!(*b, 1);

    let mut x = (1, 2);
    {
        let a = &x.0;
        let b = &mut x.1;
        *b = 5;
        assert_eq!(*a, 1);
    }
    assert_eq!(x.0, 1);
    assert_eq!(x.1, 5);


    let x = Foo(1, 2);
    let a = &x.0;
    let b = &x.0;
    assert_eq!(*a, 1);
    assert_eq!(*b, 1);

    let mut x = Foo(1, 2);
    {
        let a = &x.0;
        let b = &mut x.1;
        *b = 5;
        assert_eq!(*a, 1);
    }
    assert_eq!(x.0, 1);
    assert_eq!(x.1, 5);
}
