// skip-filecheck
//@ compile-flags: -O -C debuginfo=0 -Zmir-opt-level=2

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

// EMIT_MIR simple_option_map.map_via_question_mark.PreCodegen.after.mir
pub fn map_via_question_mark(x: Option<i32>) -> Option<i32> {
    Some(x? + 1)
}

fn main() {
    assert_eq!(None, ezmap(None));
}
