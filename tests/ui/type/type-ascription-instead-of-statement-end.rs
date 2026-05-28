fn main() {
    println!("test"): //~ ERROR statements are terminated with a semicolon
    0;
}

fn foo() {
    println!("test"): 0; //~ ERROR expected one of
}
