#![warn(clippy::needless_bitwise_bool)]
#![allow(clippy::const_is_empty)]

fn returns_bool() -> bool {
    true
}

const fn const_returns_bool() -> bool {
    false
}

fn main() {
    let (x, y) = (false, true);
    if x & y {
        println!("true")
    }
    if returns_bool() & x {
        println!("true")
    }
    if !returns_bool() & returns_bool() {
        println!("true")
    }
    if y & !x {
        println!("true")
    }

    // BELOW: lints we hope to catch as `Expr::can_have_side_effects` improves.
    if y & !const_returns_bool() {
        println!("true") // This is a const function, in an UnOp
    }

    if y & "abcD".is_empty() {
        println!("true") // This is a const method call
    }

    if y & (0 < 1) {
        println!("true") // This is a BinOp with no side effects
    }
}
