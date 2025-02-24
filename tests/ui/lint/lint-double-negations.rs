//@ check-pass
fn main() {
    let x = 1;
    -x;
    -(-x);
    --x; //~ WARN use of a double negation
    ---x; //~ WARN use of a double negation
    let _y = --(-x); //~ WARN use of a double negation
}
