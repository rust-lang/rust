//@ compile-flags: -Znext-solver
//@ check-pass

// regression test for https://github.com/rust-lang/rust/issues/163817

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
