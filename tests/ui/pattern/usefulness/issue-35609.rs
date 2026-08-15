enum Enum {
    A, B, C, D, E, F
}
use Enum::*;

struct S(Enum, ());
struct Sd { x: Enum, y: () }

fn main() {
    match (A, ()) { //~ ERROR non-exhaustive
        (A, _) => {}
    }

    match (A, A) { //~ ERROR non-exhaustive
        (_, A) => {}
    }

    match ((A, ()), ()) { //~ ERROR non-exhaustive
        ((A, ()), _) => {}
    }

    match ((A, ()), A) { //~ ERROR non-exhaustive
        ((A, ()), _) => {}
    }

    match ((A, ()), ()) { //~ ERROR non-exhaustive
        ((A, _), _) => {}
    }


    match S(A, ()) { //~ ERROR non-exhaustive
        S(A, _) => {}
    }

    match (Sd { x: A, y: () }) { //~ ERROR non-exhaustive
        Sd { x: A, y: _ } => {}
    }

    match Some(A) { //~ ERROR non-exhaustive
        Some(A) => (),
        None => ()
    }
}
