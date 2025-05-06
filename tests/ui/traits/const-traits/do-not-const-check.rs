//@ check-pass
#![feature(const_trait_impl, rustc_attrs)]

#[const_trait]
trait IntoIter {
    (const) fn into_iter(self);
}

#[const_trait]
trait Hmm: Sized {
    #[rustc_do_not_const_check]
    (const) fn chain<U>(self, other: U) where U: IntoIter,
    {
        other.into_iter()
    }
}

fn main() {}
