

#![allow(clippy)]

fn main() {
    match 1 {
        1 => {}
        2 => {
            [0; 1];
        }
        _ => {}
    }
}
