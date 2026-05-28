//@ edition:2018
//@ check-pass

#![feature(type_alias_impl_trait)]
trait Trait<'a, 'b> {}
impl<T> Trait<'_, '_> for T {}

// Here we wind up selecting `'a` and `'b` in the hidden type because
// those are the types that appear in the original values.

type Foo<'a, 'b> = impl Trait<'a, 'b>;

#[define_opaque(Foo)]
fn upper_bounds<'a, 'b>(a: &'a u8, b: &'b u8) -> Foo<'a, 'b> {
    // In this simple case, you have a hidden type `(&'0 u8, &'1 u8)` and constraints like
    //
    // ```
    // 'a: '0
    // 'b: '1
    // '0 in ['a, 'b]
    // '1 in ['a, 'b]
    // ```
    //
    // We use the fact that `'a: 0'` must hold (combined with the in
    // constraint) to determine that `'0 = 'a` must be the answer.
    (a, b)
}

fn main() {}
