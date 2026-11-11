// Test that an error on a sub-expresson in a for loop has the correct span.

fn main() {
    // Odd formatting to make sure we get the right span.
    for t in &
      foo //~ ERROR cannot find value `foo` in this scope
    {
    }
}
