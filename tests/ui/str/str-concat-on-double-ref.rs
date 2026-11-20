fn main() {
    let a: &String = &"1".to_owned();
    let b: &str = &"2";
    let c = a + b;
    //~^ ERROR: can't add `&str` to `&String`
    println!("{:?}", c);
}
