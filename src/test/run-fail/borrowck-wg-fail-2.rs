// error-pattern:borrowed

// Test that write guards trigger when there is a write to a field
// of a frozen structure.

struct S {
    x: int
}

fn main() {
    let x = @mut S { x: 3 };
    let y: &S = x;
    let z = x;
    z.x = 5;
}
