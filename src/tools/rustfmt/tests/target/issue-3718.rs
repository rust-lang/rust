fn main() {
    let x: &[i32] = &[2, 2];
    match x {
        [_a, _] => println!("Wrong username or password"),
        _ => println!("Logged in"),
    }
}
