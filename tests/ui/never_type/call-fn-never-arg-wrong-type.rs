// Test that we can't pass other types for !

#![feature(never_type)]

fn foo(x: !) -> ! {
    x
}

fn main() {
    foo("wow"); //~ ERROR mismatched types
}
