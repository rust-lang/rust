//@ revisions: current next
//@ [next] compile-flags: -Znext-solver
//@ ignore-compare-mode-next-solver (explicit revisions)

// Regression test for <https://github.com/rust-lang/rust/issues/142832>.

trait Trait {}
struct W<T>(T);
impl<T> Trait for W<W<T>>
where
    W < T > : for < 'a > Trait,
    W < T > : for < 'a > Trait,
{ }
fn func<T: Trait>() {}
fn main() {
    func::<W<_>>();
    //~^ ERROR overflow
}
