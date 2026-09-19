// FIXME(fmease): Write a description.

type DynIncompat0 = dyn Sized; //~ ERROR not dyn compatible

// FIXME(fmease): Well, if this breakage got accepted linking to this issue
//                would be moot / nonsensical. Remove the link.
// issue: <https://github.com/rust-lang/rust/issues/153731>
type DynIncompat1 = dyn HasAssocConst; //~ ERROR not dyn compatible

type DynIncompat2<'a> = dyn HasGenericAssocType<Type<()> = ()>; //~ ERROR not dyn compatible

trait HasAssocConst { const N: usize; }
trait HasGenericAssocType { type Type<T>; }

fn main() {}
