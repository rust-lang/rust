#[warn(cast_precision_loss, cast_possible_truncation, cast_sign_loss, cast_possible_wrap, cast_lossless)]
#[allow(no_effect, unnecessary_operation)]
fn main() {
    // Casting from *size
    1isize as i8;
    1isize as f64;
    1usize as f64;
    1isize as f32;
    1usize as f32;
    1isize as i32;
    1isize as u32;
    1usize as u32;
    1usize as i32;
    // Casting to *size
    1i64 as isize;
    1i64 as usize;
    1u64 as isize;
    1u64 as usize;
    1u32 as isize;
    1u32 as usize; // Should not trigger any lint
    1i32 as isize; // Neither should this
    1i32 as usize;
}
