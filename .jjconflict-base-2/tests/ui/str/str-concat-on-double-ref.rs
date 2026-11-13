fn main() {
    let a: &String = &"1".to_owned();
    let b: &str = &"2";
    let c = a + b;
    //~^ ERROR cannot add `&str` to `&String`
    println!("{:?}", c);
}
