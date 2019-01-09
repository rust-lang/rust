fn main() {
    let a: &String = &"1".to_owned();
    let b: &str = &"2";
    let c = a + b;
    //~^ ERROR binary operation `+` cannot be applied to type `&std::string::String`
    println!("{:?}", c);
}
