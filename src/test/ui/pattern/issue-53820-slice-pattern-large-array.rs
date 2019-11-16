// check-pass
#![feature(slice_patterns)]

fn main() {
    const LARGE_SIZE: usize = 1024 * 1024;
    let [..] = [0u8; LARGE_SIZE];
    match [0u8; LARGE_SIZE] {
        [..] => {}
    }
}
