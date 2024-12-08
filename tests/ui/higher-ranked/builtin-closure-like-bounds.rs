//@ edition:2024
//@ revisions: current next
//@[next] compile-flags: -Znext-solver
//@ check-pass

// Makes sure that we support closure/coroutine goals where the signature of
// the item references higher-ranked lifetimes from the *predicate* binder,
// not its own internal signature binder.
//
// This was fixed in <https://github.com/rust-lang/rust/pull/122267>.

#![feature(unboxed_closures, gen_blocks)]

trait Dispatch {
    fn dispatch(self);
}

struct Fut<T>(T);
impl<T: for<'a> Fn<(&'a (),)>> Dispatch for Fut<T>
where
    for<'a> <T as FnOnce<(&'a (),)>>::Output: Future,
{
    fn dispatch(self) {
        (self.0)(&());
    }
}

struct Gen<T>(T);
impl<T: for<'a> Fn<(&'a (),)>> Dispatch for Gen<T>
where
    for<'a> <T as FnOnce<(&'a (),)>>::Output: Iterator,
{
    fn dispatch(self) {
        (self.0)(&());
    }
}

struct Closure<T>(T);
impl<T: for<'a> Fn<(&'a (),)>> Dispatch for Closure<T>
where
    for<'a> <T as FnOnce<(&'a (),)>>::Output: Fn<(&'a (),)>,
{
    fn dispatch(self) {
        (self.0)(&())(&());
    }
}

fn main() {
    async fn foo(_: &()) {}
    Fut(foo).dispatch();

    gen fn bar(_: &()) {}
    Gen(bar).dispatch();

    fn uwu<'a>(x: &'a ()) -> impl Fn(&'a ()) { |_| {} }
    Closure(uwu).dispatch();
}
