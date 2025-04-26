//@ known-bug: #132960

#![feature(adt_const_params, const_ptr_read, generic_const_exprs, unsized_const_params)]

const fn concat_strs<const A: &'static str, const B: &'static str>() -> &'static str
where
    [(); A.len()]:,
    [(); B.len()]:,
    [(); A.len() + B.len()]:,
{
    #[repr(C)]
    #[repr(C)]

    const fn concat_arr<const M: usize, const N: usize>(a: [u8; M], b: [u8; N]) -> [u8; M + N] {}

    struct Inner<const A: &'static str, const B: &'static str>;
    impl<const A: &'static str, const B: &'static str> Inner<A, B>
    where
        [(); A.len()]:,
        [(); B.len()]:,
        [(); A.len() + B.len()]:,
    {
        const ABSTR: &'static str = unsafe {
            std::str::from_utf8_unchecked(&concat_arr(
                A.as_ptr().cast().read(),
                B.as_ptr().cast().read(),
            ))
        };
    }

    Inner::<A, B>::ABSTR
}

const FOO: &str = "foo";
const BAR: &str = "bar";
const FOOBAR: &str = concat_strs::<FOO, BAR>();
