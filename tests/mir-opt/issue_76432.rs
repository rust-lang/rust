// skip-filecheck
// EMIT_MIR_FOR_EACH_PANIC_STRATEGY
// Check that we do not insert StorageDead at each target if StorageDead was never seen

use std::fmt::Debug;

// EMIT_MIR issue_76432.test.SimplifyComparisonIntegral.diff
fn test<T: Copy + Debug + PartialEq>(x: T) {
    let v: &[T] = &[x, x, x];
    match v {
        [ref v1, ref v2, ref v3] => [v1 as *const _, v2 as *const _, v3 as *const _],
        _ => unreachable!(),
    };
}

fn main() {
    test(0u32);
}
