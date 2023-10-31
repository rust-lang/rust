// EMIT_MIR_FOR_EACH_PANIC_STRATEGY
//! Tests that we bail out when there are multiple assignments to the same local.
// unit-test: CopyProp
fn val() -> i32 {
    1
}

fn cond() -> bool {
    true
}

// EMIT_MIR branch.foo.CopyProp.diff
fn foo() -> i32 {
    let x = val();

    let y = if cond() {
        x
    } else {
        val();
        x
    };

    y
}

fn main() {
    foo();
}
