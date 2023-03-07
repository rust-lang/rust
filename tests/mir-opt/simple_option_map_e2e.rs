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

// EMIT_MIR simple_option_map_e2e.ezmap.PreCodegen.after.mir
pub fn ezmap(x: Option<i32>) -> Option<i32> {
    map(x, |n| n + 1)
}

fn main() {
    assert_eq!(None, ezmap(None));
}
