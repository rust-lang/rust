// pp-exact

// Check that `let x: _ = 0;` does not print as `let x = 0;`.

fn main() {
    let x: _ = 0;

    let _ = x;
}
