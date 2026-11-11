//! Test that deeply nested generic traits with complex bounds
//! don't cause excessive memory usage during type checking.
//!
//! Regression test for <https://github.com/rust-lang/rust/issues/31849>.

//@ run-pass

pub trait Upcast<T> {
    fn upcast(self) -> T;
}

impl<S1, S2, T1, T2> Upcast<(T1, T2)> for (S1, S2)
where
    S1: Upcast<T1>,
    S2: Upcast<T2>,
{
    fn upcast(self) -> (T1, T2) {
        (self.0.upcast(), self.1.upcast())
    }
}

impl Upcast<()> for () {
    fn upcast(self) -> () {
        ()
    }
}

pub trait ToStatic {
    type Static: 'static;
    fn to_static(self) -> Self::Static
    where
        Self: Sized;
}

impl<T, U> ToStatic for (T, U)
where
    T: ToStatic,
    U: ToStatic,
{
    type Static = (T::Static, U::Static);
    fn to_static(self) -> Self::Static {
        (self.0.to_static(), self.1.to_static())
    }
}

impl ToStatic for () {
    type Static = ();
    fn to_static(self) -> () {
        ()
    }
}

trait Factory {
    type Output;
    fn build(&self) -> Self::Output;
}

impl<S, T> Factory for (S, T)
where
    S: Factory,
    T: Factory,
    S::Output: ToStatic,
    <S::Output as ToStatic>::Static: Upcast<S::Output>,
{
    type Output = (S::Output, T::Output);
    fn build(&self) -> Self::Output {
        (self.0.build().to_static().upcast(), self.1.build())
    }
}

impl Factory for () {
    type Output = ();
    fn build(&self) -> Self::Output {
        ()
    }
}

fn main() {
    // Deeply nested tuple to trigger the original performance issue
    let it = ((((((((((), ()), ()), ()), ()), ()), ()), ()), ()), ());
    it.build();
}
