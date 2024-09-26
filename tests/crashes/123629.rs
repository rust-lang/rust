//@ known-bug: #123629
#![feature(generic_assert)]

fn foo()
where
    for<const N: usize = { assert!(u) }> ():,
{
}

fn main() {}
