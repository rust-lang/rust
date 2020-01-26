#![deny(irrefutable_let_patterns)]

fn main() {
    if let _ = 5 {} //~ ERROR irrefutable `let` pattern

    while let _ = 5 {
        //~^ ERROR irrefutable `let` pattern
        break;
    }
}
