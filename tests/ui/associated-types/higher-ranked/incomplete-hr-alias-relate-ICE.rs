//@ revisions: old next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver
//@[old] failure-status: 101
//@[old] dont-check-compiler-stderr
//@[old] known-bug: #140211
//@[next] check-pass

trait Id {
    type This<'a>;
}

#[derive(Copy, Clone)]
struct W<T>(T);
impl Id for W<u32> {
    type This<'a> = W<u32>;
}
impl Id for W<i32> {
    type This<'a> = W<i32>;
}

trait Trait<T> {}
// when using this impl we only normalize after instantiating
// with infer vars, at this point `<?t as Id>::This<'a>` is still
// ambig, so we keep the alias around and require it to structurally
// relate.
impl<T: Id> Trait<for<'a> fn(T::This<'a>)> for T {}
fn is_trait<T: Trait<U>, U>(_x: T) {}

fn main() {
    let x = W(1);
    // We prove `W<_>: Trait<for<'a> fn(<W<_>>::This<'a>)>` during
    // typeck, and `W<i32>: Trait<for<'a> fn(<W<i32>)>` during borrowck.
    // This then fails.
    is_trait(x);
    let _: W<i32> = x;
}
