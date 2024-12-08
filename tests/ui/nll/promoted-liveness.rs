// Test that promoted that have larger mir bodies than their containing function
// don't cause an ICE.

//@ check-pass

fn main() {
    &["0", "1", "2", "3", "4", "5", "6", "7"];
}
