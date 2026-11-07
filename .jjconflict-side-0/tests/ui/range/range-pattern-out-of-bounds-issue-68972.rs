#![allow(unreachable_patterns)]
fn main() {
    match 0u8 {
        251..257 => {}
        //~^ ERROR literal out of range
        251..=256 => {}
        //~^ ERROR literal out of range
        _ => {}
    }
}
