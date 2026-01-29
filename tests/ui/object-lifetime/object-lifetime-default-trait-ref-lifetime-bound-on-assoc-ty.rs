// Check that "children (assoc items) can't influence the parent (trait) wrt. obj lt defs".
// More concretely, check that lifetime bounds on assoc item defs don't influence the ambient object
// lifetime defaults that are induced by the relevant trait ref.
// So "information doesn't flow backwards".

//@ check-pass

trait Trait<T: ?Sized> { type Assoc<'a> where T: 'a; }
impl<T: ?Sized> Trait<T> for () { type Assoc<'a> = &'a T where T: 'a; }

trait Bound {}

// We deduce `dyn Bound + 'static`.
fn f0(x: impl Trait<dyn Bound>) { /*check*/ g0(x) }
fn g0(_: impl Trait<dyn Bound + 'static>) {}

// We deduce `dyn Bound + 'static` (and not `dyn Bound + 'r`).
//     We intentionally don't make use of the `T: 'a` / `dyn Trait: 'r` bound
//     that has to hold for the associated type binding to be valid.
fn f1<'r>(x: impl Trait<dyn Bound, Assoc<'r> = ()>) { /*check*/ g1(x) }
fn g1(_: impl Trait<dyn Bound + 'static>) {}

// We deduce `dyn Bound + 'static` (and not `dyn Bound + 'r`).
fn f2<'r>(x: <() as Trait<dyn Bound>>::Assoc<'r>) { /*check*/ g2(x) }
fn g2<'r>(_: <() as Trait<dyn Bound + 'static>>::Assoc<'r>) {}

fn main() {}
