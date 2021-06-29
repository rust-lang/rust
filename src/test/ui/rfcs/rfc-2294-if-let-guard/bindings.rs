#![feature(if_let_guard)]
#![allow(incomplete_features)]

fn main() {
    match Some(None) {
        Some(x) if let Some(y) = x => (x, y),
        _ => y, //~ ERROR cannot find value `y`
    }
    y //~ ERROR cannot find value `y`
}
