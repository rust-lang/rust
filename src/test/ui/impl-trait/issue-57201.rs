// Regression test for #57201

#![feature(impl_trait_in_bindings)]
#![allow(incomplete_features)]

fn bug<'a, 'b, T>()
where
    'a: 'b,
{
    let f: &impl Fn(&'a T) -> &'b T = &|x| x;
    //~^ ERROR: lifetimes in impl Trait types in bindings are not currently supported
}

fn main() {}
