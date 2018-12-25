// run-pass
pub fn main() {
    // exits early if println! evaluates its arguments, otherwise it
    // will hit the panic.
    println!("{:?}", { if true { return; } });

    panic!();
}
