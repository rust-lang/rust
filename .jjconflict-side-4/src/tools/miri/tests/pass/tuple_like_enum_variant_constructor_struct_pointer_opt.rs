#[derive(Copy, Clone, PartialEq, Debug)]
struct A<'a> {
    x: i32,
    y: &'a i32,
}

#[derive(Copy, Clone, PartialEq, Debug)]
struct B<'a>(i32, &'a i32);

#[derive(Copy, Clone, PartialEq, Debug)]
enum C<'a> {
    Value(i32, &'a i32),
    #[allow(dead_code)]
    NoValue,
}

fn main() {
    let x = 5;
    let a = A { x: 99, y: &x };
    assert_eq!(Some(a).map(Some), Some(Some(a)));
    let f = B;
    assert_eq!(Some(B(42, &x)), Some(f(42, &x)));
    // the following doesn't compile :(
    //let f: for<'a> fn(i32, &'a i32) -> B<'a> = B;
    //assert_eq!(Some(B(42, &x)), Some(f(42, &x)));
    assert_eq!(B(42, &x), foo(&x, B));
    let f = C::Value;
    assert_eq!(C::Value(42, &x), f(42, &x));
}

fn foo<'a, F: Fn(i32, &'a i32) -> B<'a>>(i: &'a i32, f: F) -> B<'a> {
    f(42, i)
}
