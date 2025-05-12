//@ run-pass
#![feature(box_patterns)]

#[derive(Clone)]
enum Noun
{
    Atom(isize),
    Cell(Box<Noun>, Box<Noun>)
}

fn fas(n: &Noun) -> Noun
{
    match n {
        &Noun::Cell(box Noun::Atom(2), box Noun::Cell(ref a, _)) => (**a).clone(),
        _ => panic!("Invalid fas pattern")
    }
}

pub fn main() {
    fas(
        &Noun::Cell(Box::new(Noun::Atom(2)),
        Box::new(Noun::Cell(Box::new(Noun::Atom(2)), Box::new(Noun::Atom(3)))))
    );
}
