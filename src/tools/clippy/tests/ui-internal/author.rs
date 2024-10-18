#![warn(clippy::author)]

fn main() {
    #[clippy::author]
    let x: char = 0x45 as char;
}
