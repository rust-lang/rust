use std::time::Duration;

// EMIT_MIR match_pattern_type.system_time_math.built.after.mir
fn system_time_math() {
    match Duration::ZERO {
        Duration::ZERO => {}
        _ => {}
    }
}

fn main() {}
