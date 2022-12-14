fn main() {
    let split = match Some(1) {
        Some(v) => v,
        None => return,
    };

    let _prev = Some(split);
    assert_eq!(split, 1);
}


// EMIT_MIR issue_73223.main.SimplifyArmIdentity.diff
