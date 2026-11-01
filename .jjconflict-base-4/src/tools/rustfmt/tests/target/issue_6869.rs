fn main() {
    let x = 0.5;

    match x {
        1. .. => println!("{x} >= 1"),
        _ => println!("{x} < 1"),
    }
}
