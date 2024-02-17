//@ build-pass (FIXME(62277): could be check-pass?)

#[derive(PartialEq, Eq)]
enum Cake {
    BlackForest,
    Marmor,
}
use Cake::*;

struct Pair<A, B>(A, B);

const BOO: Pair<Cake, Cake> = Pair(Marmor, BlackForest);
const FOO: Cake = BOO.1;

const fn foo() -> Cake {
    Marmor
}

const WORKS: Cake = Marmor;

const GOO: Cake = foo();

fn main() {
    match BlackForest {
        FOO => println!("hi"),
        GOO => println!("meh"),
        WORKS => println!("möp"),
        _ => println!("bye"),
    }
}
