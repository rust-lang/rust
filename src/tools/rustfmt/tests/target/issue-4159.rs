extern "C" {
    type A: Ord;

    type A<'a>
    where
        'a: 'static;

    type A<T: Ord>
    where
        T: 'static;

    type A = u8;

    type A<'a: 'static, T: Ord + 'static>: Eq + PartialEq
    where
        T: 'static + Copy,
    = Vec<u8>;
}
