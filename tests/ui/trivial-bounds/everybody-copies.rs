//! Regression test for #131507
//@ compile-flags: -Zmir-enable-passes=+GVN -Zmir-enable-passes=+Inline -Zvalidate-mir --crate-type lib
//@ build-pass

#![expect(incomplete_features)]
#![feature(non_lifetime_binders)]

fn brick()
where
    for<T> T: Copy,
{
    || format_args!("");
}
