//@ run-pass

#[derive(Debug, PartialEq)]
enum Test {
    Foo(usize),
    Bar(isize),
}

fn main() {
    let a = Test::Foo(10);
    let b = Test::Bar(-20);
    match (a, b) {
        (_, Test::Foo(_)) => unreachable!(),
        (Test::Foo(x), b) => {
            assert_eq!(x, 10);
            assert_eq!(b, Test::Bar(-20));
        }
        _ => unreachable!(),
    }
}
