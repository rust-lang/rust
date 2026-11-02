//@ revisions: current next
//[current]~^ ERROR overflow evaluating the requirement
//[next]~^^ ERROR overflow evaluating the requirement `<str as Noop>::Assoc == _`
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver

// Regression test for <https://github.com/rust-lang/rust/issues/151723>.
//
// Checking whether `NoopNewtype<str>` is well-formed used to hang
// with the new solver. The hang was fixed by #163496.

trait Noop {
    type Assoc: ?Sized;
}

impl<T: ?Sized> Noop for T
//~^ ERROR overflow evaluating the requirement
where
    <NoopNewtype<T> as Noop>::Assoc: Noop,
{
    type Assoc = T;
    //[next]~^ ERROR overflow evaluating the requirement `<NoopNewtype<T> as Noop>::Assoc == _`
}

impl<T> Noop for NoopNewtype<T> {}
//~^ ERROR conflicting implementations of trait `Noop` for type `NoopNewtype<_>`

struct NoopNewtype<T: ?Sized + Noop>(T::Assoc);
//[current]~^ ERROR overflow evaluating the requirement

fn coerce_newtype() -> NoopNewtype<str> {
    //[current]~^ ERROR overflow evaluating the requirement
    //[next]~^^ ERROR overflow evaluating the requirement `NoopNewtype<str>: Sized`
    //[next]~^^^ ERROR overflow evaluating whether `NoopNewtype<str>` is well-formed
    todo!()
}

fn main() {}
