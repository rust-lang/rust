// Check that traits with various kinds of associated items cause
// dropck to inject extra region constraints.

#![allow(non_camel_case_types)]

trait HasSelfMethod { fn m1(&self) { } }
trait HasMethodWithSelfArg { fn m2(x: &Self) { } }
trait HasType { type Something; }

impl HasSelfMethod for i32 { }
impl HasMethodWithSelfArg for i32 { }
impl HasType for i32 { type Something = (); }

impl<'a,T> HasSelfMethod for &'a T { }
impl<'a,T> HasMethodWithSelfArg for &'a T { }
impl<'a,T> HasType for &'a T { type Something = (); }

// e.g., `impl_drop!(Send, D_Send)` expands to:
//   ```rust
//   struct D_Send<T:Send>(T);
//   impl<T:Send> Drop for D_Send<T> { fn drop(&mut self) { } }
//   ```
macro_rules! impl_drop {
    ($Bound:ident, $Id:ident) => {
        struct $Id<T:$Bound>(T);
        impl <T:$Bound> Drop for $Id<T> { fn drop(&mut self) { } }
    }
}

impl_drop!{HasSelfMethod,        D_HasSelfMethod}
impl_drop!{HasMethodWithSelfArg, D_HasMethodWithSelfArg}
impl_drop!{HasType,              D_HasType}

fn f_sm() {
    let (_d, d1);
    d1 = D_HasSelfMethod(1);
    _d = D_HasSelfMethod(&d1);
}
//~^^ ERROR `d1` does not live long enough
fn f_mwsa() {
    let (_d, d1);
    d1 = D_HasMethodWithSelfArg(1);
    _d = D_HasMethodWithSelfArg(&d1);
}
//~^^ ERROR `d1` does not live long enough
fn f_t() {
    let (_d, d1);
    d1 = D_HasType(1);
    _d = D_HasType(&d1);
}
//~^^ ERROR `d1` does not live long enough

fn main() {
    f_sm();
    f_mwsa();
    f_t();
}
