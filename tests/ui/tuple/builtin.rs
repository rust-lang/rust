//@ check-pass

#![feature(tuple_trait)]

fn assert_is_tuple<T: std::marker::Tuple + ?Sized>() {}

struct Unsized([u8]);

fn from_param_env<T: std::marker::Tuple + ?Sized>() {
    assert_is_tuple::<T>();
}

fn main() {
    assert_is_tuple::<()>();
    assert_is_tuple::<(i32,)>();
    assert_is_tuple::<(Unsized,)>();
    from_param_env::<()>();
    from_param_env::<(i32,)>();
    from_param_env::<(Unsized,)>();
}
