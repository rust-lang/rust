#![feature(unsafe_binders, builtin_syntax)]
#![allow(incomplete_features)]

#[macro_export]
macro_rules! extern_unwrap {
    ($e:expr) => {
        ::std::unsafe_binder::unwrap_binder!($e)
    };
}

#[macro_export]
macro_rules! extern_builtin_unwrap {
    ($e:expr) => {
        builtin # unwrap_binder($e)
    };
}
