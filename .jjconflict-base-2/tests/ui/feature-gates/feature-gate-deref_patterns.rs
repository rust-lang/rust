fn main() {
    let x = Box::new('c');
    match x {
        'c' => (), //~ ERROR mismatched types
        _ => (),
    }
}
