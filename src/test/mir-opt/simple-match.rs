// Test that we don't generate unnecessarily large MIR for very simple matches


// EMIT_MIR simple_match.match_bool.mir_map.0.mir
fn match_bool(x: bool) -> usize {
    match x {
        true => 10,
        _ => 20,
    }
}

fn main() {}
