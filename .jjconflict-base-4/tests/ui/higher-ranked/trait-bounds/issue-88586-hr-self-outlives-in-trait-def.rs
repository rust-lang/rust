// Regression test for #88586: a higher-ranked outlives bound on Self in a trait
// definition caused an ICE when debug_assertions were enabled.
//
// Made to pass as part of fixing #98095.
//
//@ check-pass

trait A where
    for<'a> Self: 'a,
{
}

fn main() {}
