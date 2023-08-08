// check-pass
// compile-flags: -Ztrait-solver=next
#![feature(rustc_attrs)]

#[rustc_coinductive]
trait Trait<T> {}
impl<'a, 'b, T> Trait<T> for (&'a (), &'b ())
where
    'b: 'a,
    &'a (): Trait<T>,
{}

impl Trait<i32> for &'static () {}
impl<'a> Trait<u32> for &'a ()
where
    for<'b> (&'a (), &'b ()): Trait<u32>,
{}


fn impls_trait<T: Trait<U>, U>() {}

fn main() {
    // This infers to `impls_trait::<(&'static (), &'static ()), i32>();`
    //
    // In the first attempt we have 2 candidates for `&'a (): Trait<_>`
    // and we get ambiguity. The result is therefore ambiguity with a `'b: 'a`
    // constraint. The next attempt then uses that provisional result when
    // trying to apply `impl<'a> Trait<u32> for &'a ()`. This means we get a
    // `for<'b> 'b: 'a` bound which fails the leak check. Because of this we
    // end up with a single impl for `&'a (): Trait<_>` which infers `_` to `i32`
    // and succeeds.
    impls_trait::<(&(), &()), _>();
}
