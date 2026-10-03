fn main() {
    let x = 1;

    let y = 2;

    println!("x + y = {}", x + y);
}

fn foo() {
    #![attribute]

    let x = 1;

    // comment
}
// comment after item

// comment before item
fn bar() {
    let x = 1;
    // comment after statement

    // comment before statement
    let y = 2;
    let z = 3;

    println!("x + y + z = {}", x + y + z);
}
