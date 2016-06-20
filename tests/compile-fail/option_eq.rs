//error-pattern: can't handle cast: tmp0 as isize (Misc)
// no-ignore-x86 ignore-x86_64
fn main() {
    assert_eq!(std::char::from_u32('x' as u32), Some('x'));
}
