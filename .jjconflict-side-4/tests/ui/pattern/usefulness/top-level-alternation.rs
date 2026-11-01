#![deny(unreachable_patterns)]

fn main() {
    while let 0..=2 | 1 = 0 {} //~ ERROR unreachable pattern
    if let 0..=2 | 1 = 0 {} //~ ERROR unreachable pattern

    match 0u8 {
        0
            | 0 => {} //~ ERROR unreachable pattern
        _ => {}
    }
    match Some(0u8) {
        Some(0)
            | Some(0) => {} //~ ERROR unreachable pattern
        _ => {}
    }
    match (0u8, 0u8) {
        (0, _) | (_, 0) => {}
        (0, 0) => {} //~ ERROR unreachable pattern
        (1, 1) => {}
        _ => {}
    }
    match (0u8, 0u8) {
        (0, 1) | (2, 3) => {}
        (0, 3) => {}
        (2, 1) => {}
        _ => {}
    }
    match (0u8, 0u8) {
        (_, 0) | (_, 1) => {}
        _ => {}
    }
    match (0u8, 0u8) {
        (0, _) | (1, _) => {}
        _ => {}
    }
    match Some(0u8) {
        None | Some(_) => {}
        _ => {} //~ ERROR unreachable pattern
    }
    match Some(0u8) {
        None | Some(_) => {}
        Some(_) => {} //~ ERROR unreachable pattern
        None => {} //~ ERROR unreachable pattern
    }
    match Some(0u8) {
        Some(_) => {}
        None => {}
        None | Some(_) => {} //~ ERROR unreachable pattern
    }
    match 0u8 {
        1 | 2 => {},
        1..=2 => {}, //~ ERROR unreachable pattern
        _ => {},
    }
    let (0 | 0) = 0 else { return }; //~ ERROR unreachable pattern
}
