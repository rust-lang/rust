//@ check-pass

pub fn yes_vec_partial_eq_array<A, B>() -> impl PartialEq<[B; 32]>
where
    A: PartialEq<B>,
{
    Vec::<A>::new()
}

pub fn yes_vec_partial_eq_ref_array<'a, A, B>() -> impl PartialEq<&'a [B; 32]>
where
    A: PartialEq<B>,
{
    Vec::<A>::new()
}

pub fn yes_array_into_vec<T>() -> Vec<T> {
    [].into()
}

pub fn yes_array_into_box<T>() -> Box<[T]> {
    [].into()
}

use std::collections::VecDeque;

pub fn yes_vecdeque_partial_eq_array<A, B>() -> impl PartialEq<[B; 32]>
where
    A: PartialEq<B>,
{
    VecDeque::<A>::new()
}

pub fn yes_vecdeque_partial_eq_ref_array<'a, A, B>() -> impl PartialEq<&'a [B; 32]>
where
    A: PartialEq<B>,
{
    VecDeque::<A>::new()
}

pub fn yes_vecdeque_partial_eq_ref_mut_array<'a, A, B>() -> impl PartialEq<&'a mut [B; 32]>
where
    A: PartialEq<B>,
{
    VecDeque::<A>::new()
}

fn main() {}
