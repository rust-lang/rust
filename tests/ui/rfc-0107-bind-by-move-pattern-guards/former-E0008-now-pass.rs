// This test used to emit E0008 but now passed since `bind_by_move_pattern_guards`
// have been stabilized.

// check-pass

fn main() {
    match Some("hi".to_string()) {
        Some(s) if s.len() == 0 => {},
        _ => {},
    }
}
