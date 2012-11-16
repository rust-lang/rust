use cmp::Eq;

trait Hahaha: Eq, Eq, Eq, Eq, Eq, Eq, Eq, Eq, Eq, Eq, Eq, Eq, Eq, Eq, Eq, Eq, //~ ERROR Duplicate supertrait in trait declaration
              Eq, Eq, Eq, Eq, Eq, Eq, Eq, Eq, Eq, Eq, Eq, Eq, Eq, Eq, Eq, Eq,
              Eq, Eq, Eq, Eq, Eq, Eq, Eq, Eq, Eq, Eq, Eq, Eq, Eq, Eq, Eq, Eq,
              Eq, Eq, Eq, Eq, Eq, Eq, Eq, Eq, Eq, Eq, Eq, Eq, Eq, Eq, Eq, Eq,
              Eq, Eq, Eq, Eq, Eq, Eq, Eq, Eq, Eq, Eq, Eq, Eq, Eq, Eq, Eq, Eq,
              Eq, Eq, Eq, Eq, Eq, Eq, Eq, Eq, Eq, Eq, Eq, Eq, Eq, Eq, Eq, Eq {}

enum Lol = int;

pub impl Lol: Hahaha {
    pure fn eq(other: &Lol) -> bool { *self != **other }
    pure fn ne(other: &Lol) -> bool { *self == **other }
}

fn main() {
    if Lol(2) == Lol(4) {
        io::println("2 == 4");
    } else {
        io::println("2 != 4");
    }
}