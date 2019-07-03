// edition:2018
// compile-pass
// revisions: migrate mir
//[mir]compile-flags: -Z borrowck=mir

#![feature(member_constraints)]

trait Trait<'a, 'b> {}
impl<T> Trait<'_, '_> for T {}

// `Ordinary<'a> <: Ordinary<'b>` if `'a: 'b`, as with most types.
//
// I am purposefully avoiding the terms co- and contra-variant because
// their application to regions depends on how you interpreted Rust
// regions. -nikomatsakis
struct Ordinary<'a>(&'a u8);

// Here we wind up selecting `'e` in the hidden type because
// we need something outlived by both `'a` and `'b` and only `'e` applies.

fn upper_bounds<'a, 'b, 'c, 'd, 'e>(a: Ordinary<'a>, b: Ordinary<'b>) -> impl Trait<'d, 'e>
where
    'a: 'e,
    'b: 'e,
    'a: 'd,
{
    // We return a value:
    //
    // ```
    // 'a: '0
    // 'b: '1
    // '0 in ['d, 'e]
    // ```
    //
    // but we don't have it.
    //
    // We are forced to pick that '0 = 'e, because only 'e is outlived by *both* 'a and 'b.
    let p = if condition() { a } else { b };
    p
}

fn condition() -> bool {
    true
}

fn main() {}
