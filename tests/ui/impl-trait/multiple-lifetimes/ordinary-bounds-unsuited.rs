//@ edition:2018

trait Trait<'a, 'b> {}
impl<T> Trait<'_, '_> for T {}

// `Ordinary<'a> <: Ordinary<'b>` if `'a: 'b`, as with most types.
//
// I am purposefully avoiding the terms co- and contra-variant because
// their application to regions depends on how you interpreted Rust
// regions. -nikomatsakis
struct Ordinary<'a>(&'a u8);

// Here we need something outlived by `'a` *and* outlived by `'b`, but
// we can only name `'a` and `'b` (and neither suits). So we get an
// error. Somewhat unfortunate, though, since the caller would have to
// consider the loans for both `'a` and `'b` alive.

fn upper_bounds<'a, 'b>(a: Ordinary<'a>, b: Ordinary<'b>) -> impl Trait<'a, 'b>
{
    // We return a value:
    //
    // ```
    // 'a: '0
    // 'b: '1
    // '0 in ['a, 'b]
    // ```
    //
    // but we don't have it.
    //
    // We are forced to pick that '0 = 'e, because only 'e is outlived by *both* 'a and 'b.
    if condition() { a } else { b }
    //~^ ERROR hidden type for `impl Trait<'a, 'b>` captures lifetime that does not appear in bounds
}

fn condition() -> bool {
    true
}

fn main() {}
