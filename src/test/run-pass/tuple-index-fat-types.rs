struct Foo<'a>(&'a [isize]);

fn main() {
    let x: &[isize] = &[1, 2, 3];
    let y = (x,);
    assert_eq!(y.0, x);

    let x: &[isize] = &[1, 2, 3];
    let y = Foo(x);
    assert_eq!(y.0, x);
}
