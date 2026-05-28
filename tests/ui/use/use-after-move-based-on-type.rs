fn main() {
    let x = "Hello!".to_string();
    let _y = x;
    println!("{}", x); //~ ERROR borrow of moved value
}
