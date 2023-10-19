// skip-filecheck
// compile-flags: -O -C debuginfo=0 -Zmir-opt-level=2
// only-64bit

#[inline(always)]
fn map<T, U, F>(slf: Option<T>, f: F) -> Option<U>
where
    F: FnOnce(T) -> U,
{
    match slf {
        Some(x) => Some(f(x)),
        None => None,
    }
}

// EMIT_MIR simple_option_map.ezmap.PreCodegen.after.mir
pub fn ezmap(x: Option<i32>) -> Option<i32> {
    map(x, |n| n + 1)
}

fn main() {
    assert_eq!(None, ezmap(None));
}
