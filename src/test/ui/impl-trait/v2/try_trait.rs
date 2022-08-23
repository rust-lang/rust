// check-pass

#![feature(return_position_impl_trait_v2)]

pub(crate) struct NeverShortCircuit<T>(pub T);

impl<T> NeverShortCircuit<T> {
    /// Wrap a binary `FnMut` to return its result wrapped in a `NeverShortCircuit`.
    #[inline]
    pub fn wrap_mut_2<A, B>(mut f: impl FnMut(A, B) -> T) -> impl FnMut(A, B) -> Self {
        move |a, b| NeverShortCircuit(f(a, b))
    }
}

fn main() {}
