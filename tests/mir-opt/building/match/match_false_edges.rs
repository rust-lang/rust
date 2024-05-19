// skip-filecheck
fn guard() -> bool {
    false
}

fn guard2(_: i32) -> bool {
    true
}

// no_mangle to make sure this gets instantiated even in an executable.
#[no_mangle]
// EMIT_MIR match_false_edges.full_tested_match.built.after.mir
pub fn full_tested_match() {
    let _ = match Some(42) {
        Some(x) if guard() => (1, x),
        Some(y) => (2, y),
        None => (3, 3),
    };
}

// no_mangle to make sure this gets instantiated even in an executable.
#[no_mangle]
// EMIT_MIR match_false_edges.full_tested_match2.built.after.mir
pub fn full_tested_match2() {
    let _ = match Some(42) {
        Some(x) if guard() => (1, x),
        None => (3, 3),
        Some(y) => (2, y),
    };
}

// EMIT_MIR match_false_edges.main.built.after.mir
fn main() {
    let _ = match Some(1) {
        Some(_w) if guard() => 1,
        _x => 2,
        Some(y) if guard2(y) => 3,
        _z => 4,
    };
}
