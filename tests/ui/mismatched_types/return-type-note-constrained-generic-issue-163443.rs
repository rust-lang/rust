use std::mem::MaybeUninit;

fn test(x: &mut MaybeUninit<u32>) {
    x.write(1i32); //~ ERROR mismatched types
}

fn main() {}
