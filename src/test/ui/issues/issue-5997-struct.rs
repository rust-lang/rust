fn f<T>() -> bool {
    struct S(T); //~ ERROR can't use type parameters from outer function

    true
}

fn main() {
    let b = f::<isize>();
    assert!(b);
}
