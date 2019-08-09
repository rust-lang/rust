// run-pass
#![feature(box_patterns)]
#![feature(box_syntax)]

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
    fas(&Noun::Cell(box Noun::Atom(2), box Noun::Cell(box Noun::Atom(2), box Noun::Atom(3))));
}
