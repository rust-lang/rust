// Demonstrates that `impl<T> const Clone for Option<T>` does not require const_hack bounds.
// See issue #144207.
//@ revisions: next old
//@ [next] compile-flags: -Znext-solver
//@ check-pass

#![feature(const_trait_impl, const_destruct)]

use std::marker::Destruct;

#[const_trait]
pub trait CloneLike: Sized {
    fn clone(&self) -> Self;

    fn clone_from(&mut self, source: &Self)
    where
        Self: [const] Destruct,
    {
        *self = source.clone()
    }
}

pub enum OptionLike<T> {
    None,
    Some(T),
}

impl<T> const CloneLike for OptionLike<T>
where
    T: [const] CloneLike,
{
    fn clone(&self) -> Self {
        match self {
            Self::Some(x) => Self::Some(x.clone()),
            Self::None => Self::None,
        }
    }

    fn clone_from(&mut self, source: &Self)
    where
        Self: [const] Destruct,
    {
        match (self, source) {
            (Self::Some(to), Self::Some(from)) => to.clone_from(from),
            (to, from) => *to = from.clone(),
        }
    }
}

fn main() {}
