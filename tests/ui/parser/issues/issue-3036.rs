//@ run-rustfix

// Testing that semicolon tokens are printed correctly in errors

fn main() {
    let _x = 3 //~ ERROR: expected `;`
}
