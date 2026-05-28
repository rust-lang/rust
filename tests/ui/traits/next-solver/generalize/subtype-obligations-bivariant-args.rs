//@ revisions: old next
//@[next] compile-flags: -Znext-solver
//@ ignore-compare-mode-next-solver (explicit revisions)
//@ check-pass

// Regression test for trait-system-refactor-initiative#250.
// Subtyping previously handled bivariant arguments by emitting
// a `WellFormed` obligation when generalizing them.
//
// This obligation then got dropped inside of an ambiguous `Subtype`
// obligation so we never constrained the bivariant arg.

// Test case 1
enum State<S, T>
where
    S: Iterator<Item = T>,
{
    Active { upstream: S },
    WindDown,
    Complete,
}

impl<S, T> State<S, T>
where
    S: Iterator<Item = T>,
{
    fn foo(self) {
        let x = match self {
            State::Active { .. } => None,
            State::WindDown => None,
            State::Complete => Some(State::Complete),
        };
        let _: Option<State<S, T>> = x;
    }
}

// Test case 2
trait Trait {
    type Assoc;
}
impl<T> Trait for T {
    type Assoc = T;
}

struct Foo<T: Trait<Assoc = U>, U>(T);

fn main() {
    let x = None.unwrap();
    let y = x;
    let _: Foo<_,  _> = x;
    let _: Foo<u32, u32> = x;
}
