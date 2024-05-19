// With no repr attribute the discriminant will default to isize.
// On 32-bit architectures this is equivalent to i32 so the variants
// collide. On other architectures we need compilation to fail anyway,
// so force the repr.
#[cfg_attr(not(target_pointer_width = "32"), repr(i32))]
enum Eu64 {
    //~^ ERROR discriminant value `0` assigned more than once
    Au64 = 0,
    //~^NOTE `0` assigned here
    Bu64 = 0x8000_0000_0000_0000
    //~^NOTE `0` (overflowed from `9223372036854775808`) assigned here
}

fn main() {}
