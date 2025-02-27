//@ compile-flags: -O -C debuginfo=0 -Zmir-opt-level=2

#[inline]
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
    // We expect this to all be inlined, as though it was written without the
    // combinator and without the closure, using just a plain match.

    // CHECK-LABEL: fn ezmap
    // CHECK: [[INNER:_.+]] = copy ((_1 as Some).0: i32);
    // CHECK: [[SUCC:_.+]] = Add({{copy|move}} [[INNER]], const 1_i32);
    // CHECK: _0 = Option::<i32>::Some({{copy|move}} [[SUCC]]);
    map(x, |n| n + 1)
}

// EMIT_MIR simple_option_map.map_via_question_mark.PreCodegen.after.mir
pub fn map_via_question_mark(x: Option<i32>) -> Option<i32> {
    // FIXME(#138544): Ideally this would optimize out the `ControlFlow` local.

    // CHECK-LABEL: fn map_via_question_mark
    // CHECK: [[INNER:_.+]] = copy ((_1 as Some).0: i32);
    // CHECK: [[TEMP1:_.+]] = ControlFlow::<Option<Infallible>, i32>::Continue(copy [[INNER]]);
    // CHECK: [[TEMP2:_.+]] = copy (([[TEMP1]] as Continue).0: i32);
    // CHECK: [[SUCC:_.+]] = Add({{copy|move}} [[TEMP2]], const 1_i32);
    // CHECK: _0 = Option::<i32>::Some({{copy|move}} [[SUCC]]);
    Some(x? + 1)
}

fn main() {
    assert_eq!(None, ezmap(None));
    assert_eq!(None, map_via_question_mark(None));
}
