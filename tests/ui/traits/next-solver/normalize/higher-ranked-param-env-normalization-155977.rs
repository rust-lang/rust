//@ compile-flags: -Znext-solver

// Regression test for #155977: the new solver used to ICE while normalizing
// a higher-ranked where-clause in the param env.
// Fixed by eager param-env normalization in #156976.

trait Trait<Input> {
    type Output;
}

fn walk2<F>()
//~^ ERROR the trait bound
where
    for<'a> F: 'a,
    for<'a> <F as Trait<&'a u32>>::Output: 'a,
{
}

fn main() {}
