trait A<const B: bool> {}

//     vv- Let's call this const "UNEVALUATED" for the comment below.
impl A<{}> for () {}
//~^ ERROR mismatched types

// During overlap check, we end up trying to prove `(): A<?0c>`. Inference guides
// `?0c = UNEVALUATED` (which is the `{}` const in the erroneous impl). We then
// fail to prove `ConstArgHasType<UNEVALUATED, u8>` since `UNEVALUATED` has the
// type `bool` from the type_of query. We then deeply normalize the predicate for
// error reporting, which ends up normalizing `UNEVALUATED` to a ConstKind::Error.
// This ended up ICEing when trying to report an error for the `ConstArgHasType`
// predicate, since we don't expect `ConstArgHasType(ERROR, Ty)` to ever fail.

trait C<const D: u8> {}
impl<const D: u8> C<D> for () where (): A<D> {}
impl<const D: u8> C<D> for () {}

fn main() {}
