fn loop_ending() -> i32 {
    loop {
        if false { break; } //~ ERROR mismatched types
        return 42;
    }
}

fn main() {}
