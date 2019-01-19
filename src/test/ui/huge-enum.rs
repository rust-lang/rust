// normalize-stderr-test "std::option::Option<\[u32; \d+\]>" -> "TYPE"
// normalize-stderr-test "\[u32; \d+\]" -> "TYPE"

#[cfg(target_pointer_width = "32")]
fn main() {
    let big: Option<[u32; (1<<29)-1]> = None;
}

#[cfg(target_pointer_width = "64")]
fn main() {
    let big: Option<[u32; (1<<45)-1]> = None;
}
