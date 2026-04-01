//@ run-rustfix

fn main() {
    match Some(1) { //~ ERROR non-exhaustive patterns: `None` not covered
        Some(1) => {}
        // hello
        Some(_) => {}
    }
}
