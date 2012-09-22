// xfail-test
// error-pattern:ran out of stack
fn main() {
    eat(move ~0);
}

fn eat(
    +a: ~int
) {
    eat(move a)
}
