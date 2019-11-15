// build-pass

// Checks that the compiler does not actually try to allocate 4 TB during compilation and OOM crash.

#[cfg(target_pointer_width = "64")]
const SIZE: usize = 4 * 1024 * 1024 * 1024 * 1024;
#[cfg(target_pointer_width = "32")]
const SIZE: usize = 2 * 1024 * 1024 * 1024;

fn main() {
    [0; SIZE];
}
