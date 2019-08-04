// run-rustfix

#![warn(clippy::unnecessary_cast)]
#![allow(clippy::no_effect, clippy::unnecessary_operation)]

fn main() {
    // casting integer literal to float is unnecessary
    100 as f32;
    100 as f64;
    100_i32 as f64;
    // Should not trigger
    #[rustfmt::skip]
    let v = vec!(1);
    &v as &[i32];
    1.0 as f64;
    1 as u64;
}
