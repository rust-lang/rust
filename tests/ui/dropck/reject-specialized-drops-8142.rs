// Issue 8142: Test that Drop impls cannot be specialized beyond the
// predicates attached to the type definition itself.
trait Bound {
    fn foo(&self) {}
}
struct K<'l1, 'l2> {
    x: &'l1 i8,
    y: &'l2 u8,
}
struct L<'l1, 'l2> {
    x: &'l1 i8,
    y: &'l2 u8,
}
struct M<'m> {
    x: &'m i8,
}
struct N<'n> {
    x: &'n i8,
}
struct O<To> {
    x: *const To,
}
struct P<Tp> {
    x: *const Tp,
}
struct Q<Tq> {
    x: *const Tq,
}
struct R<Tr> {
    x: *const Tr,
}
struct S<Ts: Bound> {
    x: *const Ts,
}
struct T<'t, Ts: 't> {
    x: &'t Ts,
}
struct U;
struct V<Tva, Tvb> {
    x: *const Tva,
    y: *const Tvb,
}
struct W<'l1, 'l2> {
    x: &'l1 i8,
    y: &'l2 u8,
}
struct X<const Ca: usize>;
struct Y<const Ca: usize, const Cb: usize>;

enum Enum<T> {
    Variant(T),
}
struct TupleStruct<T>(T);
union Union<T: Copy> {
    f: T,
}

impl<'al, 'adds_bnd: 'al> Drop for K<'al, 'adds_bnd> {
    //~^ ERROR `Drop` impl requires `'adds_bnd: 'al`
    fn drop(&mut self) {}
}

impl<'al, 'adds_bnd> Drop for L<'al, 'adds_bnd>
//~^ ERROR `Drop` impl requires `'adds_bnd: 'al`
where
    'adds_bnd: 'al,
{
    fn drop(&mut self) {}
}

impl<'ml> Drop for M<'ml> {
    fn drop(&mut self) {}
}

impl Drop for N<'static> {
    //~^ ERROR `Drop` impls cannot be specialized
    fn drop(&mut self) {}
}

impl<COkNoBound> Drop for O<COkNoBound> {
    fn drop(&mut self) {}
}

impl Drop for P<i8> {
    //~^ ERROR `Drop` impls cannot be specialized
    fn drop(&mut self) {}
}

impl<AddsBnd: Bound> Drop for Q<AddsBnd> {
    //~^ ERROR `Drop` impl requires `AddsBnd: Bound`
    fn drop(&mut self) {}
}

impl<'rbnd, AddsRBnd: 'rbnd> Drop for R<AddsRBnd> {
    fn drop(&mut self) {}
}

impl<Bs: Bound> Drop for S<Bs> {
    fn drop(&mut self) {}
}

impl<'t, Bt: 't> Drop for T<'t, Bt> {
    fn drop(&mut self) {}
}

impl Drop for U {
    fn drop(&mut self) {}
}

impl<One> Drop for V<One, One> {
    //~^ ERROR `Drop` impls cannot be specialized
    fn drop(&mut self) {}
}

impl<'lw> Drop for W<'lw, 'lw> {
    //~^ ERROR `Drop` impls cannot be specialized
    fn drop(&mut self) {}
}

impl Drop for X<3> {
    //~^ ERROR `Drop` impls cannot be specialized
    fn drop(&mut self) {}
}

impl<const Ca: usize> Drop for Y<Ca, Ca> {
    //~^ ERROR `Drop` impls cannot be specialized
    fn drop(&mut self) {}
}

impl<AddsBnd: Bound> Drop for Enum<AddsBnd> {
    //~^ ERROR `Drop` impl requires `AddsBnd: Bound`
    fn drop(&mut self) {}
}

impl<AddsBnd: Bound> Drop for TupleStruct<AddsBnd> {
    //~^ ERROR `Drop` impl requires `AddsBnd: Bound`
    fn drop(&mut self) {}
}

impl<AddsBnd: Copy + Bound> Drop for Union<AddsBnd> {
    //~^ ERROR `Drop` impl requires `AddsBnd: Bound`
    fn drop(&mut self) {}
}

pub fn main() {}
