// Check that we correctly deduce object lifetime defaults inside self types of qualified paths.

//@ check-pass

trait Outer { type Ty; }
trait Inner {}

impl<'a> Outer for dyn Inner + 'a { type Ty = &'a (); }

// We deduce `dyn Inner + 'static` from absence of any bounds on self ty param of trait `Outer`.
//
// Prior to PR rust-lang/rust#129543, assoc tys weren't considered *eligible generic containers* and
// thus we'd use the *ambient object lifetime default* induced by the reference type ctor `&`,
// namely `'r`. Now however, the assoc ty shadows/overwrites that ambient default to `'static`.
fn f<'r>(x: &'r <dyn Inner as Outer>::Ty) { /*check*/ g(x) }
fn g<'r>(x: &'r <dyn Inner + 'static as Outer>::Ty) {}

fn main() {}
