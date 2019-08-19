// edition:2018
// run-pass
// revisions: migrate mir
//[mir]compile-flags: -Z borrowck=mir

#![feature(member_constraints)]

trait Trait<'a, 'b> {}
impl<T> Trait<'_, '_> for T {}

// `Invert<'a> <: Invert<'b>` if `'b: 'a`, unlike most types.
//
// I am purposefully avoiding the terms co- and contra-variant because
// their application to regions depends on how you interpreted Rust
// regions. -nikomatsakis
struct Invert<'a>(fn(&'a u8));

fn upper_bounds<'a, 'b, 'c, 'd, 'e>(a: Invert<'a>, b: Invert<'b>) -> impl Trait<'d, 'e>
where
    'c: 'a,
    'c: 'b,
    'd: 'c,
{
    // Representing the where clauses as a graph, where `A: B` is an
    // edge `B -> A`:
    //
    // ```
    // 'a -> 'c -> 'd
    //        ^
    //        |
    //       'b
    // ```
    //
    // Meanwhile we return a value &'0 u8 where we have the constraints:
    //
    // ```
    // '0: 'a
    // '0: 'b
    // '0 in ['d, 'e]
    // ```
    //
    // Here, ignoring the "in" constraint, the minimal choice for `'0`
    // is `'c`, but that is not in the "in set". Still, that reduces
    // the range of options in the "in set" to just `'d` (`'e: 'c`
    // does not hold).
    let p = if condition() { a } else { b };
    p
}

fn condition() -> bool {
    true
}

fn main() {}
