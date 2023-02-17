#![feature(non_lifetime_binders)]
//~^ WARN is incomplete and may not be safe

pub fn foo()
where
    for<V> V: Sized,
{
}

fn main() {
    foo();
    //~^ ERROR the size for values of type `V` cannot be known at compilation time
}
