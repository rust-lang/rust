// issue: <https://github.com/rust-lang/rust/issues/11552>
// Test nested box pattern matching inside a larger `match` statement.
//@ run-pass
#![feature(deref_patterns)]

#[derive(Clone)]
enum Noun {
    Atom(isize),
    Cell(Box<Noun>, Box<Noun>),
}

fn fas(n: &Noun) -> Noun {
    match n {
        &Noun::Cell(Noun::Atom(2), Noun::Cell(ref a, _)) => (**a).clone(),
        _ => panic!("Invalid fas pattern"),
    }
}

pub fn main() {
    fas(&Noun::Cell(
        Box::new(Noun::Atom(2)),
        Box::new(Noun::Cell(Box::new(Noun::Atom(2)), Box::new(Noun::Atom(3)))),
    ));
}
