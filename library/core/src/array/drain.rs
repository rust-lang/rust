use crate::assert_unsafe_precondition;
use crate::marker::Destruct;
use crate::mem::ManuallyDrop;

#[rustc_const_unstable(feature = "array_try_map", issue = "79711")]
#[unstable(feature = "array_try_map", issue = "79711")]
pub(super) struct Drain<'a, T, U, const N: usize, F: FnMut(T) -> U> {
    array: ManuallyDrop<[T; N]>,
    moved: usize,
    f: &'a mut F,
}
#[rustc_const_unstable(feature = "array_try_map", issue = "79711")]
#[unstable(feature = "array_try_map", issue = "79711")]
impl<T, U, const N: usize, F> const FnOnce<(usize,)> for &mut Drain<'_, T, U, N, F>
where
    F: [const] FnMut(T) -> U,
{
    type Output = U;

    extern "rust-call" fn call_once(mut self, args: (usize,)) -> Self::Output {
        self.call_mut(args)
    }
}
#[rustc_const_unstable(feature = "array_try_map", issue = "79711")]
#[unstable(feature = "array_try_map", issue = "79711")]
impl<T, U, const N: usize, F> const FnMut<(usize,)> for &mut Drain<'_, T, U, N, F>
where
    F: [const] FnMut(T) -> U,
{
    extern "rust-call" fn call_mut(&mut self, (i,): (usize,)) -> Self::Output {
        // SAFETY: increment moved before moving. if `f` panics, we drop the rest.
        self.moved += 1;
        assert_unsafe_precondition!(
            check_library_ub,
            "musnt index array out of bounds", (i: usize = i, size: usize = N) => i < size
        );
        // SAFETY: caller guarantees never called with number >= N (see `Drain::new`)
        (self.f)(unsafe { self.array.as_ptr().add(i).read() })
    }
}
#[rustc_const_unstable(feature = "array_try_map", issue = "79711")]
#[unstable(feature = "array_try_map", issue = "79711")]
impl<T: [const] Destruct, U, const N: usize, F: FnMut(T) -> U> const Drop
    for Drain<'_, T, U, N, F>
{
    fn drop(&mut self) {
        let mut n = self.moved;
        while n != N {
            // SAFETY: moved must always be < N
            unsafe { self.array.as_mut_ptr().add(n).drop_in_place() };
            n += 1;
        }
    }
}
impl<'a, T, U, const N: usize, F: FnMut(T) -> U> Drain<'a, T, U, N, F> {
    /// SAFETY: must be called without indexing out of bounds. (see `Drain::call_mut`)
    // FIXME(const-hack): this is a hack for `let guard = Guard(array); |i| f(guard[i])`.
    pub(super) const unsafe fn new(array: [T; N], f: &'a mut F) -> Self {
        Self { array: ManuallyDrop::new(array), moved: 0, f }
    }
}
