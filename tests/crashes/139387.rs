//@ known-bug: #139387
//@ needs-rustc-debug-assertions

trait A {
    fn method() -> impl Sized;
}
trait B {
    fn method(Hash: Wrap<impl Beta<U: Copy + for<'a> Epsilon<'_, SI1: Eta>>>) -> impl Sized;
}

fn ambiguous<T: A + B>()
where
    T::method(..): Send,
{
}
