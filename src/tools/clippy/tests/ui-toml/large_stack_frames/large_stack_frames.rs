#![warn(clippy::large_stack_frames)]

// We use this helper function instead of writing [0; 4294967297] directly to represent a
// case that large_stack_arrays can't catch
fn create_array<const N: usize>() -> [u8; N] {
    [0; N]
}

fn f() {
    let _x = create_array::<1000>();
}
fn f2() {
    //~^ ERROR: this function allocates a large amount of stack space
    let _x = create_array::<1001>();
}

fn main() {}
