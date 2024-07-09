// skip-filecheck
// Test that we don't generate unnecessarily large MIR for very simple matches

// EMIT_MIR simple_match.match_bool.built.after.mir
fn match_bool(x: bool) -> usize {
    match x {
        true => 10,
        _ => 20,
    }
}

pub enum E1 {
    V1,
    V2,
    V3,
}

// EMIT_MIR simple_match.match_enum.built.after.mir
pub fn match_enum(x: E1) -> bool {
    match x {
        E1::V1 | E1::V2 => true,
        E1::V3 => false,
    }
}

fn main() {}
