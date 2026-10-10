//@ check-pass

// Test the parsing of half-open ranges.

fn main() {}

#[cfg(false)]
fn syntax() {
    match scrutinee {
        X.. | 0.. | 'a'.. | 0.0f32.. => {}
        ..=X | ..X => {}
        ..=0 | ..0 => {}
        ..='a' | ..'a' => {}
        ..=0.0f32 | ..0.0f32 => {}
    }
}

fn syntax2() {
    macro_rules! mac {
        ($e:expr) => {
            match 0u8 { ..$e => {}, _ => {} }
            match 0u8 { ..=$e => {}, _ => {} }
            match 0u8 { $e.. => {}, _ => {} }
        }
    }
    mac!(42u8);
}
