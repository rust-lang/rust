// Test that we can't pass other types for !

fn foo(x: !) -> ! {
    x
}

fn main() {
    foo("wow"); //~ ERROR mismatched types
}
