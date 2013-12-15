// error-pattern:borrowed

// Test that write guards trigger when there is a write to a directly
// frozen @mut box.

#[feature(managed_boxes)];

fn main() {
    let x = @mut 3;
    let _y: &mut int = x;
    let z = x;
    *z = 5;
}
