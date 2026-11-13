// Verifies that the `locals` command lists the names of user-declared variables
// in the current stack frame. After stepping into `main` we should see `x` and `y`.
fn main() {
    let x = 1_i32;
    let y = true;
    let _ = (x, y);
}
