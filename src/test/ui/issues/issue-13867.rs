// run-pass
// Test that codegen works correctly when there are multiple refutable
// patterns in match expression.


enum Foo {
    FooUint(usize),
    FooNullary,
}

fn main() {
    let r = match (Foo::FooNullary, 'a') {
        (Foo::FooUint(..), 'a'..='z') => 1,
        (Foo::FooNullary, 'x') => 2,
        _ => 0
    };
    assert_eq!(r, 0);

    let r = match (Foo::FooUint(0), 'a') {
        (Foo::FooUint(1), 'a'..='z') => 1,
        (Foo::FooUint(..), 'x') => 2,
        (Foo::FooNullary, 'a') => 3,
        _ => 0
    };
    assert_eq!(r, 0);

    let r = match ('a', Foo::FooUint(0)) {
        ('a'..='z', Foo::FooUint(1)) => 1,
        ('x', Foo::FooUint(..)) => 2,
        ('a', Foo::FooNullary) => 3,
        _ => 0
    };
    assert_eq!(r, 0);

    let r = match ('a', 'a') {
        ('a'..='z', 'b') => 1,
        ('x', 'a'..='z') => 2,
        _ => 0
    };
    assert_eq!(r, 0);

    let r = match ('a', 'a') {
        ('a'..='z', 'b') => 1,
        ('x', 'a'..='z') => 2,
        ('a', 'a') => 3,
        _ => 0
    };
    assert_eq!(r, 3);
}
