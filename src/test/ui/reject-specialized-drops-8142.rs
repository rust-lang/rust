// Issue 8142: Test that Drop impls cannot be specialized beyond the
// predicates attached to the struct/enum definition itself.

trait Bound { fn foo(&self) { } }
struct K<'l1,'l2> { x: &'l1 i8, y: &'l2 u8 }
struct L<'l1,'l2> { x: &'l1 i8, y: &'l2 u8 }
struct M<'m> { x: &'m i8 }
struct N<'n> { x: &'n i8 }
struct O<To> { x: *const To }
struct P<Tp> { x: *const Tp }
struct Q<Tq> { x: *const Tq }
struct R<Tr> { x: *const Tr }
struct S<Ts:Bound> { x: *const Ts }
struct T<'t,Ts:'t> { x: &'t Ts }
struct U;
struct V<Tva, Tvb> { x: *const Tva, y: *const Tvb }
struct W<'l1, 'l2> { x: &'l1 i8, y: &'l2 u8 }

impl<'al,'adds_bnd:'al> Drop for K<'al,'adds_bnd> {                        // REJECT
    //~^ ERROR The requirement `'adds_bnd : 'al` is added only by the Drop impl.
    fn drop(&mut self) { } }

impl<'al,'adds_bnd>     Drop for L<'al,'adds_bnd> where 'adds_bnd:'al {    // REJECT
    //~^ ERROR The requirement `'adds_bnd : 'al` is added only by the Drop impl.
    fn drop(&mut self) { } }

impl<'ml>               Drop for M<'ml>         { fn drop(&mut self) { } } // ACCEPT

impl                    Drop for N<'static>     { fn drop(&mut self) { } } // REJECT
//~^ ERROR mismatched types
//~| expected type `N<'n>`
//~|    found type `N<'static>`

impl<COkNoBound> Drop for O<COkNoBound> { fn drop(&mut self) { } } // ACCEPT

impl              Drop for P<i8>          { fn drop(&mut self) { } } // REJECT
//~^ ERROR Implementations of Drop cannot be specialized

impl<AddsBnd:Bound> Drop for Q<AddsBnd> { fn drop(&mut self) { } } // REJECT
//~^ ERROR The requirement `AddsBnd: Bound` is added only by the Drop impl.

impl<'rbnd,AddsRBnd:'rbnd> Drop for R<AddsRBnd> { fn drop(&mut self) { } } // REJECT
//~^ ERROR The requirement `AddsRBnd : 'rbnd` is added only by the Drop impl.

impl<Bs:Bound>    Drop for S<Bs>          { fn drop(&mut self) { } } // ACCEPT

impl<'t,Bt:'t>    Drop for T<'t,Bt>       { fn drop(&mut self) { } } // ACCEPT

impl              Drop for U              { fn drop(&mut self) { } } // ACCEPT

impl<One>         Drop for V<One,One>     { fn drop(&mut self) { } } // REJECT
//~^ ERROR Implementations of Drop cannot be specialized

impl<'lw>         Drop for W<'lw,'lw>     { fn drop(&mut self) { } } // REJECT
//~^ ERROR cannot infer an appropriate lifetime

pub fn main() { }
