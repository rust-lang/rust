// Make sure we don't treat missing associated items as rigid
// during coherence, even if we know they've got an impossible
// `Sized`-bound. As we check whether the self type is definitely
// not `Sized` outside of coherence, this check can be incomplete.
//
// In this test we only use `impl<T> Overlap<u32> for T` to normalize
// the field of `MaybeUnsized<T, u32>` when checking whether it's
// definitely not `Sized`. However, for `MaybeUnsized<u32, u32>` we
// could also use `impl<U> Overlap<U> for u32` for normalization, which
// would result in a `Sized` type. cc #139000

struct MaybeUnsized<T: Overlap<U>, U>(<T as Overlap<U>>::MaybeUnsized);

trait ReqSized {
    type Missing1
    where
        Self: Sized;
    type Missing2
    where
        Self: Sized;
}
impl<T> ReqSized for MaybeUnsized<T, u32> {}

struct W<T: ?Sized>(T);
trait Eq<T> {}
impl<T> Eq<T> for W<T> {}

trait RelateReqSized {}
impl<T: ReqSized> RelateReqSized for T where W<T::Missing1>: Eq<T::Missing2> {}

trait Overlap<U> {
    type MaybeUnsized: ?Sized;
}
impl<T> Overlap<u32> for T {
    type MaybeUnsized = str;
}
impl<U> Overlap<U> for u32
//~^ ERROR conflicting implementations of trait `Overlap<u32>` for type `u32`
where
    MaybeUnsized<U, u32>: RelateReqSized,
{
    type MaybeUnsized = u32;
}

fn main() {}
