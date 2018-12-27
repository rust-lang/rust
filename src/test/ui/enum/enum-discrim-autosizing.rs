// With no repr attribute the discriminant will default to isize.
// On 32-bit architectures this is equivalent to i32 so the variants
// collide. On other architectures we need compilation to fail anyway,
// so force the repr.
#[cfg_attr(not(target_pointer_width = "32"), repr(i32))]
enum Eu64 {
    Au64 = 0,
    Bu64 = 0x8000_0000_0000_0000 //~ERROR already exists
}

fn main() {}
