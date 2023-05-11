// aux-build:issue-19452-aux.rs

extern crate issue_19452_aux;

enum Homura {
    Madoka { age: u32 }
}

fn main() {
    let homura = Homura::Madoka;
    //~^ ERROR expected value, found struct variant `Homura::Madoka`

    let homura = issue_19452_aux::Homura::Madoka;
    //~^ ERROR expected value, found struct variant `issue_19452_aux::Homura::Madoka`
}
