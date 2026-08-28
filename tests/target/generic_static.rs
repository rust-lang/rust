#![feature(generic_const_items)]
pub const SORT<T: Copy + 'static, const SET: &'static [T]>: &[T] = const { 4 };

pub const SORT<
    AAAAAAAAAAAAAAAAAAAA,
    BBBBBBBBBBBBBBBBBBBb,
    CCCCCCCCCCCCCCCCCCC,
    DDDDDDDDDDDDDDDDDDDDD,
    EEEEEEEEEEEE,
    FFFFF,
    G,
>: &[T] = const { 4 };
