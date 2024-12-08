//@ known-bug: #131507
//@ compile-flags: -Zmir-opt-level=5 -Zvalidate-mir
#![feature(non_lifetime_binders)]

fn brick()
where
    for<T> T: Copy,
{
    || format_args!("");
}
