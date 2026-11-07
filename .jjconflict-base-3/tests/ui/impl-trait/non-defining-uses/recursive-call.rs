//@ revisions: current next
//@[next] compile-flags: -Znext-solver
//@ ignore-compare-mode-next-solver (explicit revisions)
//@ check-pass

// Regression test for the non-defining use error in `gll`.

struct Foo;
impl Foo {
    fn recur(&self, b: bool) -> impl Sized + '_ {
        if b {
            let temp = Foo;
            temp.recur(false);
            // desugars to `Foo::recur(&temp);`
        }

        self
    }

    fn in_closure(&self) -> impl Sized + '_ {
        let _ = || {
            let temp = Foo;
            temp.in_closure();
            // desugars to `Foo::in_closure(&temp);`
        };

        self
    }
}
fn main() {}
