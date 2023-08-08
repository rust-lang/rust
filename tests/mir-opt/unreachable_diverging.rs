// EMIT_MIR_FOR_EACH_PANIC_STRATEGY
pub enum Empty {}

fn empty() -> Option<Empty> {
    None
}

fn loop_forever() {
    loop {}
}

// EMIT_MIR unreachable_diverging.main.UnreachablePropagation.diff
fn main() {
    let x = true;
    if let Some(bomb) = empty() {
        if x {
            loop_forever()
        }
        match bomb {}
    }
}
