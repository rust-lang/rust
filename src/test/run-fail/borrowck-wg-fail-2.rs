// error-pattern:borrowed

// Test that write guards trigger when there is a write to a field
// of a frozen structure.

#[feature(managed_boxes)];

struct S {
    x: int
}

fn main() {
    let x = @mut S { x: 3 };
    let _y: &S = x;
    let z = x;
    z.x = 5;
}
