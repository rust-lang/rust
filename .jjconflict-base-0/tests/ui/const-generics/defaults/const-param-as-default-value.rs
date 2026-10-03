//@ run-pass
struct Foo<const N: usize, const M: usize = N>([u8; N], [u8; M]);

fn foo<const N: usize>() -> Foo<N> {
    let x = [0; N];
    Foo(x, x)
}

// To check that we actually apply the correct substs for const param defaults.
fn concrete_foo() -> Foo<13> {
    Foo(Default::default(), Default::default())
}


fn main() {
    let val = foo::<13>();
    assert_eq!(val.0, val.1);

    let val = concrete_foo();
    assert_eq!(val.0, val.1);
}
