// Ensure that we don't enter infinite recursion and trigger a stack overflow when computing the
// variances of items that reference diverging free alias types. This once used to happen in a dev
// version of PR #141030.
//
// This test demonstrates that we cannot rely on wfck bailing out early with a normalization error
// for such free alias types before we reach variance computation. At least at the time of writing,
// we wfck item `First` first which includes variance computation -- at which point item `Second`
// hasn't been wfck'ed yet.
// This means we can't use `type_of` to recurse into free alias types, we do have to use
// `expand_free_alias_tys`.

#![feature(checked_type_aliases)]

// the (unused) type parameter is necessary to actually trigger variance computation for `First`.
struct First<T>(Second);
//~^ ERROR type parameter `T` is never used
//~| ERROR overflow normalizing the type alias `Second`

type Second = Second; // diverging free alias type
//~^ ERROR overflow normalizing the type alias `Second`

// The same applies when a diverging alias mentions a type parameter. The argument grows with
// every step, so there is no repeated `(DefId, args)` pair to detect the divergence with.
// Variance inference must stop at the recursion limit, and reporting the unused parameter
// must not restart the expansion.
type Diverging<T> = Diverging<(T,)>;
//~^ ERROR overflow normalizing the type alias `Diverging<

struct Wrap<T>(Diverging<T>);
//~^ ERROR type parameter `T` is never used
//~| ERROR overflow normalizing the type alias `Diverging<

fn main() {}
