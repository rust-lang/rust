// Don't suggest removing a semicolon if the last statement isn't an expression with semicolon
// (#81098)
fn wat() -> impl core::fmt::Display { //~ ERROR: `()` doesn't implement `std::fmt::Display`
    fn why() {}
}

// Do it if the last statement is an expression with semicolon
// (#54771)
fn ok() -> impl core::fmt::Display { //~ ERROR: `()` doesn't implement `std::fmt::Display`
    1;
}

fn main() {}
