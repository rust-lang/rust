// Ideally, given an assoc type binding `dyn Trait<AssocTy = Ty>`, we'd factor in the item bounds of
// assoc type `AssocTy` when computing the ambient object lifetime default for type `Ty`.
//
// However, since the current implementation can't handle this we instead conservatively and hackily
// treat the ambient object lifetime default of the RHS as indeterminate if any lifetime arguments
// are passed to (the trait ref or) the GAT thus rejecting any hidden object lifetime bounds.
// This way, we can still implement the desired behavior in the future.

trait Outer {
    type A<'r>: ?Sized;
    type B<'r>: ?Sized + 'r;
}

trait Inner {}

// FIXME: Ideally, we'd elaborate `dyn Inner` to `dyn Inner + 'static` here instead of rejecting it.
fn f0<'r>(x: impl Outer<A<'r> = dyn Inner>) { /*check*/ g0(x) }
//~^ ERROR cannot be deduced from context
fn g0<'r>(_: impl Outer<A<'r> = dyn Inner + 'static>) {}

fn f1<'r>(x: impl Outer<B<'r> = dyn Inner + 'r>) { /*check*/ g1(x) }
// FIXME: Ideally, we'd elaborate `dyn Inner` to `dyn Inner + 'r` here instead of rejecting it.
fn g1<'r>(_: impl Outer<B<'r> = dyn Inner>) {}
//~^ ERROR cannot be deduced from context

fn main() {}
