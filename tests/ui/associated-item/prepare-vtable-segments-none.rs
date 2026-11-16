//! Regression test for <https://github.com/rust-lang/rust/issues/153368>.

//@ revisions: old next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver

trait Foo: Bar<T> + Bar<u32> {}
//~^ ERROR cannot find type `T` in this scope
trait Bar<T> {
    fn bar(self) -> T;
}

fn test_infer_version(x: &dyn Foo) {
    x.bar()
    //[next]~^ ERROR cannot move a value of type `dyn Foo`
    //[next]~| ERROR cannot move out of `*x` which is behind a shared reference
}

fn main() {}
