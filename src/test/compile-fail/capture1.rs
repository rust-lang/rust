// -*- rust -*-

// error-pattern: attempted dynamic environment-capture

fn main() {
    let bar: int = 5;
    fn foo() -> int { ret bar; }
}
