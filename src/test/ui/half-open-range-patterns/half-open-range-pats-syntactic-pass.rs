// check-pass

// Test the parsing of half-open ranges.

#![feature(exclusive_range_pattern)]
#![feature(half_open_range_patterns)]

fn main() {}

#[cfg(FALSE)]
fn syntax() {
    match scrutinee {
        X.. | 0.. | 'a'.. | 0.0f32.. => {}
        ..=X | ...X | ..X => {}
        ..=0 | ...0 | ..0 => {}
        ..='a' | ...'a' | ..'a' => {}
        ..=0.0f32 | ...0.0f32 | ..0.0f32 => {}
    }

    macro_rules! mac {
        ($e:expr) => {
            let ..$e;
            let ...$e;
            let ..=$e;
            let $e..;
            let $e...;
            let $e..=;
        }
    }

    mac!(0);
}
