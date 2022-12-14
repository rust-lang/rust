#![feature(or_patterns)]

fn main() {
    // Initialize test constants in a way that cannot be determined at compile time, to ensure
    // rustc and LLVM cannot optimize out statements (or coverage counters) downstream from
    // dependent conditions.
    let is_true = std::env::args().len() == 1;

    let mut a: u8 = 0;
    let mut b: u8 = 0;
    if is_true {
        a = 2;
        b = 0;
    }
    match (a, b) {
        // Or patterns generate MIR `SwitchInt` with multiple targets to the same `BasicBlock`.
        // This test confirms a fix for Issue #79569.
        (0 | 1, 2 | 3) => {}
        _ => {}
    }
    if is_true {
        a = 0;
        b = 0;
    }
    match (a, b) {
        (0 | 1, 2 | 3) => {}
        _ => {}
    }
    if is_true {
        a = 2;
        b = 2;
    }
    match (a, b) {
        (0 | 1, 2 | 3) => {}
        _ => {}
    }
    if is_true {
        a = 0;
        b = 2;
    }
    match (a, b) {
        (0 | 1, 2 | 3) => {}
        _ => {}
    }
}
