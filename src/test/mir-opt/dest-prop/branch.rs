//! Tests that assignment in both branches of an `if` are eliminated.

fn val() -> i32 {
    1
}

fn cond() -> bool {
    true
}

// EMIT_MIR branch.main.DestinationPropagation.diff
fn main() {
    let x = val();

    let y = if cond() {
        x
    } else {
        val();
        x
    };
}
