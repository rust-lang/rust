pub fn foo<T>() -> &'static isize {
    if false {
        static a: isize = 4;
        return &a;
    } else {
        static a: isize = 5;
        return &a;
    }
}

pub fn bar() -> &'static isize {
    foo::<isize>()
}
