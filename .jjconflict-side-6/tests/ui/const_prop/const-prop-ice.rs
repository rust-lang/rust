//@ build-fail

fn main() {
    [0; 3][3u64 as usize]; //~ ERROR this operation will panic at runtime
}
