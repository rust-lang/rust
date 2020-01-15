#![warn(clippy::match_single_binding)]
#[allow(clippy::many_single_char_names)]

fn main() {
    let a = 1;
    let b = 2;
    let c = 3;
    // Lint
    match (a, b, c) {
        (x, y, z) => {
            println!("{} {} {}", x, y, z);
        },
    }
    // Ok
    match a {
        2 => println!("2"),
        _ => println!("Not 2"),
    }
    // Ok
    let d = Some(5);
    match d {
        Some(d) => println!("5"),
        _ => println!("None"),
    }
}
