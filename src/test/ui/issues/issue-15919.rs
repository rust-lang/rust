// error-pattern: too big for the current architecture
// normalize-stderr-test "\[usize; \d+\]" -> "[usize; N]"

#[cfg(target_pointer_width = "32")]
fn main() {
    let x = [0usize; 0xffff_ffff];
}

#[cfg(target_pointer_width = "64")]
fn main() {
    let x = [0usize; 0xffff_ffff_ffff_ffff];
}
