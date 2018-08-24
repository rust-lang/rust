#![feature(slice_patterns)]

fn main() {
    let x: &[u32] = &[];
    let &[[ref _a, ref _b..]..] = x; //~ ERROR refutable pattern
}
