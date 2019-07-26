// run-pass
// Test a rather underspecified example:


pub fn main() {
    let f = {|i| i};
    assert_eq!(f(2), 2);
    assert_eq!(f(5), 5);
}
