fn main() {
    let x;

    match true {
        true => x = 42,
        false => println!("{x}") //~ ERROR E0381
    }
}
