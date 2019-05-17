#![deny(unused_parens)]

#[derive(Eq, PartialEq)]
struct X { y: bool }
impl X {
    fn foo(&self, conjunct: bool) -> bool { self.y && conjunct }
}

fn foo() -> isize {
    return (1); //~ ERROR unnecessary parentheses around `return` value
}
fn bar(y: bool) -> X {
    return (X { y }); //~ ERROR unnecessary parentheses around `return` value
}

fn main() {
    foo();
    bar((true)); //~ ERROR unnecessary parentheses around function argument

    if (true) {} //~ ERROR unnecessary parentheses around `if` condition
    while (true) {} //~ ERROR unnecessary parentheses around `while` condition
    match (true) { //~ ERROR unnecessary parentheses around `match` head expression
        _ => {}
    }
    if let 1 = (1) {} //~ ERROR unnecessary parentheses around `let` head expression
    while let 1 = (2) {} //~ ERROR unnecessary parentheses around `let` head expression
    let v = X { y: false };
    // struct lits needs parens, so these shouldn't warn.
    if (v == X { y: true }) {}
    if (X { y: true } == v) {}
    if (X { y: false }.y) {}

    while (X { y: false }.foo(true)) {}
    while (true | X { y: false }.y) {}

    match (X { y: false }) {
        _ => {}
    }

    X { y: false }.foo((true)); //~ ERROR unnecessary parentheses around method argument

    let mut _a = (0); //~ ERROR unnecessary parentheses around assigned value
    _a = (0); //~ ERROR unnecessary parentheses around assigned value
    _a += (1); //~ ERROR unnecessary parentheses around assigned value
}
