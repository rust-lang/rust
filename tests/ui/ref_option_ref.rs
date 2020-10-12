#![warn(clippy::ref_option_ref)]

fn main() {
    let x: &Option<&u32> = &None;
}
