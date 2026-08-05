//@ build-fail

fn main() {
    [1][0u64 as usize];
    [1][1.5 as usize]; //~ ERROR operation will panic
    [1][1u64 as usize]; //~ ERROR operation will panic
}
