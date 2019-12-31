// In this regression test for #67146, we check that the
// negative outlives bound `!'a` is rejected by the parser.
// This regression was first introduced in PR #57364.

fn main() {}

fn f1<T: !'static>() {}
//~^ ERROR negative bounds are not supported
fn f2<'a, T: Ord + !'a>() {}
//~^ ERROR negative bounds are not supported
fn f3<'a, T: !'a + Ord>() {}
//~^ ERROR negative bounds are not supported
