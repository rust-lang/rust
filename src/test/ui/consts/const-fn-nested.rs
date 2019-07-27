// run-pass
// Test a call whose argument is the result of another call.

const fn sub(x: u32, y: u32) -> u32 {
    x - y
}

const X: u32 = sub(sub(88, 44), 22);

fn main() {
    assert_eq!(X, 22);
}
