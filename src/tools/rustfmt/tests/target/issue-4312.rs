// issue 4312
fn main() {
    /* " */
    println!("Hello, world!");
    /* abc " */
    println!("Hello, world!");
    /* " abc */
    println!("Hello, world!");
    let y = 4;
    let x = match 1 + y == 3 {
        True => 3,
        False => 4,
        /* " unreachable */
    };
}

// issue 4806
enum X {
    A,
    B,
    /*"*/
}
