//@ compile-flags: -Znext-solver
//@ check-pass

// Regression test for https://github.com/rust-lang/rust/issues/163817
// We previously didn't normalize term from assumption and returned term
// containing non-rigid aliases as normalized result.

trait App {
    type Step<'z>;
}
trait Func<Arg> {}

fn replay<A, CB>(build_callback: CB)
where
    A: App,
    CB: Fn() -> Box<dyn for<'z> Func<A::Step<'z>>>,
{
    build_callback();
}

fn main() {}
