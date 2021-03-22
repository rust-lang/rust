//! Test for `||` in or-patterns

fn main() {
    let x = 3;

    match x {
        1 | 2 || 3 => (), //~ ERROR unexpected token `||` in pattern
        _ => (),
    }

    match x {
        (1 | 2 || 3) => (), //~ ERROR unexpected token `||` in pattern
        _ => (),
    }

    match (x,) {
        (1 | 2 || 3,) => (), //~ ERROR unexpected token `||` in pattern
        _ => (),
    }

    struct TS(u8);

    match TS(x) {
        TS(1 | 2 || 3) => (), //~ ERROR unexpected token `||` in pattern
        _ => (),
    }

    struct NS { f: u8 }

    match (NS { f: x }) {
        NS { f: 1 | 2 || 3 } => (), //~ ERROR unexpected token `||` in pattern
        _ => (),
    }

    match [x] {
        [1 | 2 || 3] => (), //~ ERROR unexpected token `||` in pattern
        _ => (),
    }

    match x {
        || 1 | 2 | 3 => (), //~ ERROR unexpected token `||` in pattern
        _ => (),
    }
}
