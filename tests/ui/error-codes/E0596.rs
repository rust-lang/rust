//@ check-pass
fn main() {
    let x = 1;
    let y = &mut x; //~ WARNING [E0596]
}
