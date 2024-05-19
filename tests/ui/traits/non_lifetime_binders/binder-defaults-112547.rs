#![feature(non_lifetime_binders)]
//~^ WARN the feature `non_lifetime_binders` is incomplete

pub fn bar()
where
    for<const N: usize = {
    (||1usize)()
}> V: IntoIterator
//~^^^ ERROR defaults for generic parameters are not allowed in `for<...>` binders
//~^^ ERROR cannot find type `V` in this scope
{
}

fn main() {
    bar();
}
