// check-pass

#![feature(return_position_impl_trait_v2)]

pub trait Try {
    type Output;
}

fn filter_try_fold<'a>(predicate: &'a mut impl FnMut(&u32) -> bool) -> impl Sized {
    22
}

fn main() {}
