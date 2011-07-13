// Checking that the compiler reports multiple type errors at once
// error-pattern:mismatched types: expected bool
// error-pattern:mismatched types: expected int

fn main() {
    let bool a = 1;
    let int b = true;
}