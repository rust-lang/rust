fn main() {
    let x = 3;
    match x {
        1 | 2 || 3 => (), //~ ERROR unexpected token `||` after pattern
        _ => (),
    }
}
