#![feature(coverage_attribute)]
//@ edition: 2021

// Regression test for inconsistent handling of function signature spans that
// are followed by code using the `?` operator.
//
// For each of these similar functions, the line containing the function
// signature should be handled in the same way.

fn a() -> Option<i32>
//
{
    Some(7i32);
    Some(0)
}

fn b() -> Option<i32>
//
{
    Some(7i32)?;
    Some(0)
}

fn c() -> Option<i32>
//
{
    let _ = Some(7i32)?;
    Some(0)
}

fn d() -> Option<i32>
//
{
    let _: () = ();
    Some(7i32)?;
    Some(0)
}

#[coverage(off)]
fn main() {
    a();
    b();
    c();
    d();
}
