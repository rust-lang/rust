// Regression test for #63033. The scenario here is:
//
// - The returned future captures the `Box<dyn T>`, which is shorthand for `Box<dyn T + 'static>`.
// - The actual value that gets captured is `Box<dyn T + '?0>` where `'static: '?0`
// - We generate a member constraint `'?0 member ['a, 'b, 'static]`
// - None of those regions are a "least choice", so we got stuck
//
// After the fix, we now select `'static` in cases where there are no upper bounds (apart from
// 'static).
//
// edition:2018
// check-pass

#![allow(dead_code)]
trait T {}
struct S;
impl S {
    async fn f<'a, 'b>(_a: &'a S, _b: &'b S, _c: Box<dyn T>) {}
}
fn main() {}
