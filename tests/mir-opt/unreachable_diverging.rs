// unit-test: UnreachablePropagation-initial
// ignore-wasm32 compiled with panic=abort by default
pub enum Empty {}

fn empty() -> Option<Empty> {
    None
}

fn loop_forever() {
    loop {}
}

// EMIT_MIR unreachable_diverging.main.UnreachablePropagation-initial.diff
fn main() {
    let x = true;
    if let Some(bomb) = empty() {
        if x {
            loop_forever()
        }
        match bomb {}
    }
}
