//@ edition:2018

trait Trait<'a, 'b> {}
impl<T> Trait<'_, '_> for T {}

// `Ordinary<'a> <: Ordinary<'b>` if `'a: 'b`, as with most types.
//
// I am purposefully avoiding the terms co- and contra-variant because
// their application to regions depends on how you interpreted Rust
// regions. -nikomatsakis
struct Ordinary<'a>(&'a u8);

// Here we get an error because none of our choices (either `'d` nor `'e`) are outlived
// by both `'a` and `'b`.

fn upper_bounds<'a, 'b, 'c, 'd, 'e>(a: Ordinary<'a>, b: Ordinary<'b>) -> impl Trait<'d, 'e>
where
    'a: 'e,
    'b: 'd,
{
    // Hidden type `Ordinary<'0>` with constraints:
    //
    // ```
    // 'a: '0
    // 'b: '0
    // 'a in ['d, 'e]
    // ```
    if condition() { a } else { b }
    //~^ ERROR hidden type for `impl Trait<'d, 'e>` captures lifetime that does not appear in bounds
}

fn condition() -> bool {
    true
}

fn main() {}
