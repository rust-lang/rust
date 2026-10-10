fn main() {
    let x = 1u8;
    match x {
        0u8..=3i8 => (), //~ ERROR E0308
        _ => ()
    }
}
