//@ compile-flags: -Znext-solver

struct NewSolver;
struct OldSolver;

fn foo<T>()
where
    T: Iterator<Item = NewSolver>,
    OldSolver: Into<T::Item>,
    //~^ ERROR the trait bound `NewSolver: From<OldSolver>` is not satisfied
{
    let x: OldSolver = OldSolver.into();
}

fn main() {}
