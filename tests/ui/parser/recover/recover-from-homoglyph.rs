fn main() {
    println!(""); //~ ERROR unknown start of token: \u{37e}
    let x: usize = (); //~ ERROR mismatched types
}
