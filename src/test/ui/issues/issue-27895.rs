fn main() {
    let i = 5;
    let index = 6;

    match i {
        0..=index => println!("winner"),
        //~^ ERROR runtime values cannot be referenced in patterns
        _ => println!("hello"),
    }
}
