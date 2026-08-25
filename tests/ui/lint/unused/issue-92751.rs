#[deny(unused)]
pub fn broken(x: Option<()>) -> i32 {
    match x {
        Some(()) => (1), //~ ERROR unnecessary parentheses around match arm expression
        None => (2), //~ ERROR unnecessary parentheses around match arm expression
    }
}

fn main() { }
