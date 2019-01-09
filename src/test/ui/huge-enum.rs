// error-pattern: Option
// normalize-stderr-test "<\[u32; \d+\]>" -> "<[u32; N]>"

// FIXME: work properly with higher limits

#[cfg(target_pointer_width = "32")]
fn main() {
    let big: Option<[u32; (1<<29)-1]> = None;
}

#[cfg(target_pointer_width = "64")]
fn main() {
    let big: Option<[u32; (1<<45)-1]> = None;
}
