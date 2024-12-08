//@ run-pass
#![feature(box_patterns)]

#[derive(Debug, PartialEq)]
enum Test {
    Foo(usize),
    Bar(isize),
}

fn main() {
    let a = Box::new(Test::Foo(10));
    let b = Box::new(Test::Bar(-20));
    match (a, b) {
        (_, box Test::Foo(_)) => unreachable!(),
        (box Test::Foo(x), b) => {
            assert_eq!(x, 10);
            assert_eq!(b, Box::new(Test::Bar(-20)));
        },
        _ => unreachable!(),
    }
}
