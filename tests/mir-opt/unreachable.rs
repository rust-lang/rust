// EMIT_MIR_FOR_EACH_PANIC_STRATEGY
enum Empty {}

fn empty() -> Option<Empty> {
    None
}

// EMIT_MIR unreachable.main.UnreachablePropagation.diff
fn main() {
    if let Some(_x) = empty() {
        let mut _y;

        if true {
            _y = 21;
        } else {
            _y = 42;
        }

        match _x { }
    }
}
