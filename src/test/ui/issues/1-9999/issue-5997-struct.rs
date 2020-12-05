fn f<T>() -> bool {
    struct S(T); //~ ERROR can't use generic parameters from outer function

    true
}

fn main() {
    let b = f::<isize>();
    assert!(b);
}
