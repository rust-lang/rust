// Regression test for #63033. The scenario here is:
//
// - The returned future captures the &'static String`
// - The actual value that gets captured is `&'?0 String` where `'static: '?0`
// - We generate a member constraint `'?0 member ['a, 'b, 'static]`
// - None of those regions are a "least choice", so we got stuck
//
// After the fix, we now select `'static` in cases where there are no upper bounds (apart from
// 'static).
//
// edition:2018
// check-pass

async fn test<'a, 'b>(test: &'a String, test2: &'b String, test3: &'static String) {}

fn main() {}
