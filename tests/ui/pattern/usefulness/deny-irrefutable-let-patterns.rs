#![feature(if_let_guard)]

#![deny(irrefutable_let_patterns)]

fn main() {
    if let _ = 5 {} //~ ERROR irrefutable `if let` pattern

    while let _ = 5 { //~ ERROR irrefutable `while let` pattern
        break;
    }

    match 5 {
        _ if let _ = 2 => {} //~ ERROR irrefutable `if let` guard pattern
        _ => {}
    }
}
