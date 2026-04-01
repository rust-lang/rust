//@ check-pass
//@ compile-flags: -Znext-solver

// See https://github.com/rust-lang/trait-system-refactor-initiative/issues/1,
// a minimization of a pattern in core.

trait Iterator {
    type Item;
}

struct Flatten<I>(I);

impl<I, U> Iterator for Flatten<I>
where
    I: Iterator<Item = U>,
{
    type Item = U;
}

fn needs_iterator<I: Iterator>() {}

fn environment<J>()
where
    J: Iterator,
{
    needs_iterator::<Flatten<J>>();
}

fn main() {}
