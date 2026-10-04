pub fn foo<T>() -> isize {
    static a: isize = 3;
    a
}

pub fn bar() -> isize {
    foo::<isize>()
}
