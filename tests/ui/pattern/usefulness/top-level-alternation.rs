#![deny(unreachable_patterns)]

fn main() {
    while let 0..=2 | 1 = 0 {}
    //~^ ERROR unreachable pattern
    //~| this pattern is unreachable
    if let 0..=2 | 1 = 0 {}
    //~^ ERROR unreachable pattern
    //~| this pattern is unreachable
    match 0u8 {
        0
            | 0 => {}
            //~^ ERROR unreachable pattern
            //~| this pattern is unreachable
        _ => {}
    }
    match Some(0u8) {
        Some(0)
            | Some(0) => {}
        //~^ ERROR unreachable pattern
        //~| this pattern is unreachable
        _ => {}
    }
    match (0u8, 0u8) {
        (0, _) | (_, 0) => {}
        (0, 0) => {}
        //~^ ERROR unreachable pattern
        //~| this arm is never executed
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
        _ => {}
        //~^ ERROR unreachable pattern
        //~| this arm is never executed
    }
    match Some(0u8) {
        None | Some(_) => {}
        Some(_) => {}
        //~^ ERROR multiple unreachable patterns
        //~| this arm is never executed
        None => {}
    }
    match Some(0u8) {
        Some(_) => {}
        None => {}
        None | Some(_) => {}
        //~^ ERROR unreachable pattern
        //~| this arm is never executed
    }
    match 0u8 {
        1 | 2 => {},
        1..=2 => {},
        //~^ ERROR unreachable pattern
        //~| this arm is never executed
        _ => {},
    }
    let (0 | 0) = 0 else { return };
    //~^ ERROR unreachable pattern
    //~| this pattern is unreachable
}
