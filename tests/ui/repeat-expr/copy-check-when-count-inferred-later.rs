// Test that we enforce repeat expr element types are `Copy` even
// when the repeat count is only inferred at a later point in type
// checking.

fn main() {
    let a = [String::new(); _];
    //~^ ERROR: the trait bound `String: Copy` is not satisfied
    let b: [_; 2] = a;
}
