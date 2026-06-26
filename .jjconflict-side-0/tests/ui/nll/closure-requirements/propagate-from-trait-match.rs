// Test that regions which appear only in the closure's generics (in
// this case, `'a`) are properly mapped to the creator's generics. In
// this case, the closure constrains its type parameter `T` to outlive
// the same `'a` for which it implements `Trait`, which can only be the `'a`
// from the function definition.

//@ compile-flags:-Zverbose-internals

#![feature(rustc_attrs)]
#![allow(dead_code)]

trait Trait<'a> {}

fn establish_relationships<T, F>(value: T, closure: F)
where
    F: FnOnce(T),
{
    closure(value)
}

fn require<'a, T>(t: T)
where
    T: Trait<'a> + 'a,
{
}

#[rustc_regions]
fn supply<'a, T>(value: T)
where
    T: Trait<'a>,
{
    establish_relationships(value, |value| {
        // This function call requires that
        //
        // (a) T: Trait<'a>
        //
        // and
        //
        // (b) T: 'a
        //
        // The latter does not hold.

        require(value);
        //~^ ERROR the parameter type `T` may not live long enough
    });
}

fn main() {}
