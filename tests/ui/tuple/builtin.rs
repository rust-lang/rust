//@ check-pass

#![feature(tuple_trait)]

fn assert_is_tuple<T: std::marker::Tuple + ?Sized>() {}

fn from_param_env<T: std::marker::Tuple + ?Sized>() {
    assert_is_tuple::<T>();
}

fn main() {
    assert_is_tuple::<()>();
    assert_is_tuple::<(i32,)>();
    from_param_env::<()>();
    from_param_env::<(i32,)>();
}
