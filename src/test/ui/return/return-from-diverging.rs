// Test that return another type in place of ! raises a type mismatch.

fn fail() -> ! {
    return "wow"; //~ ERROR mismatched types
}

fn main() {
}
