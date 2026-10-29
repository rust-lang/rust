#![feature(generic_assert)]

fn foo()
where
    for<const N: usize = { assert!(u) }> ():,
    //~^ ERROR cannot find value `u` in this scope
    //~^^ ERROR only lifetime parameters can be used in this context
    //~^^^ ERROR defaults for generic parameters are not allowed in `for<...>` binders
{
}

fn main() {}
