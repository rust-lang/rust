fn main() {
    let a: String = &String::from("a");
    //~^ ERROR mismatched types
    let b: String = &format!("b");
    //~^ ERROR mismatched types
}
