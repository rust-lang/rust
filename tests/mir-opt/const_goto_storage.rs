// unit-test: ConstGoto

// EMIT_MIR const_goto_storage.match_nested_if.ConstGoto.diff
fn match_nested_if() -> bool {
    let val = match () {
        () if if if if true { true } else { false } { true } else { false } {
            true
        } else {
            false
        } =>
            {
                true
            }
        _ => false,
    };
    val
}

fn main() {
    let _ = match_nested_if();
}
