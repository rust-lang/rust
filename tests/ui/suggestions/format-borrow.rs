fn main() {
    let a: String = &String::from("a");
    //~^ ERROR mismatched types
    let b: String = &format!("b");
    //~^ ERROR mismatched types
    let c: String = &mut format!("c");
    //~^ ERROR mismatched types
    let d: String = &mut (format!("d"));
    //~^ ERROR mismatched types
}
