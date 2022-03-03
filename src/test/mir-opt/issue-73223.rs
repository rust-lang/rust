fn main() {
    let split = match Some(1) {
        Some(v) => v,
        None => return,
    };

    let _prev = Some(split);
    assert_eq!(split, 1);
}

// EMIT_MIR_FOR_EACH_BIT_WIDTH
// These tests were broken by changes to enum deaggregation, and will be fixed when
// `SimplifyArmIdentity` is fixed more generally
// FIXME(JakobDegen) EMIT_MIR issue_73223.main.SimplifyArmIdentity.diff
// FIXME(JakobDegen) EMIT_MIR issue_73223.main.PreCodegen.diff
